import gradio as gr
import gc
import inspect
import os
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from utils import recover_image, resize_and_crop
from immunization import ImmunizationConfig, immunize_image
import numpy as np
import sys, traceback

gr.close_all()

DEFAULT_INPAINT_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"
EDITOR_IMAGE_SIZE = 512


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


INPAINT_MODEL_SOURCE = os.getenv("PHOTOGUARD_INPAINT_MODEL", DEFAULT_INPAINT_MODEL_ID)
INPAINT_LOCAL_FILES_ONLY = _env_flag("PHOTOGUARD_LOCAL_FILES_ONLY", True)
PIPELINE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DTYPE = torch.float16 if PIPELINE_DEVICE == "cuda" else torch.float32

try:
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_SOURCE,
        local_files_only=INPAINT_LOCAL_FILES_ONLY,
        torch_dtype=PIPELINE_DTYPE,
        safety_checker=None,
    )
except (OSError, ValueError) as exc:
    raise RuntimeError(
        "Unable to load the inpainting model. Set PHOTOGUARD_INPAINT_MODEL to a local directory "
        "or Hugging Face repo id. If the model is not cached locally, set "
        "PHOTOGUARD_LOCAL_FILES_ONLY=0 to allow a download. "
        f"Current source: {INPAINT_MODEL_SOURCE!r}."
    ) from exc

pipe_inpaint = pipe_inpaint.to(PIPELINE_DEVICE)
if PIPELINE_DEVICE == "cpu":
    pipe_inpaint.enable_attention_slicing()

## Good params for editing that we used all over the paper --> decent quality and speed   
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100
DEFAULT_SEED = 1234
DEFAULT_IMMUNIZATION_CONFIG = ImmunizationConfig(
    profile_name="stable_diffusion",
    target_mode="random",
    iters=20,
    target_strength=1.0,
    chaos_strength=0.35,
    denoiser_strength=0.2,
    denoiser_steps=1,
    eot_samples=1,
    resize_jitter=0.05,
    noise_strength=0.01,
    blur_kernel_size=3,
)


def _cleanup_runtime_memory():
    gc.collect()
    if PIPELINE_DEVICE == "cuda":
        torch.cuda.empty_cache()


def _make_image_mask():
    params = inspect.signature(gr.ImageMask).parameters
    kwargs = {
        "label": "Drawing tool to mask regions you want to keep, e.g. faces",
        "type": "pil",
        "image_mode": "RGB",
        "sources": ["upload"],
        "format": "webp",
        "width": EDITOR_IMAGE_SIZE,
        "height": EDITOR_IMAGE_SIZE,
    }

    # Keep the editor canvas fixed so large uploads do not balloon browser memory usage.
    if "canvas_size" in params:
        kwargs["canvas_size"] = (EDITOR_IMAGE_SIZE, EDITOR_IMAGE_SIZE)
    if "fixed_canvas" in params:
        kwargs["fixed_canvas"] = True
    if "layers" in params:
        kwargs["layers"] = False
    if "transforms" in params:
        kwargs["transforms"] = ()

    return gr.ImageMask(**kwargs)

def to_pil_image(image_input):
    """
    Normalize Gradio image input into a PIL.Image.
    Handles:
      - dicts like {'image': numpy_array} or {'image': PIL.Image}
      - numpy arrays
      - PIL.Image objects
      - bytes (rare)
    """
    # dict-like input (older Gradio versions or specific components)
    if isinstance(image_input, dict):
        # try the common keys
        for k in ("image", "data", "img"):
            if k in image_input:
                image_input = image_input[k]
                break

    # If already a PIL image, return it
    if isinstance(image_input, Image.Image):
        return image_input

    # If it's a numpy array
    if isinstance(image_input, np.ndarray):
        # if dtype is float in [0,1], convert to uint8
        if issubclass(image_input.dtype.type, np.floating):
            arr = (np.clip(image_input, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = image_input.astype(np.uint8)
        # if grayscale -> convert to RGB
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")
        return Image.fromarray(arr)

    # If bytes, try to open
    if isinstance(image_input, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(image_input)).convert("RGB")

    # Last resort: try to coerce via PIL
    try:
        return Image.fromarray(np.array(image_input)).convert("RGB")
    except Exception:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

def run(image, prompt, seed, guidance_scale, num_inference_steps, immunize=False):
    """
    Robust Gradio handler: accepts dict/tuple/plain-image payloads and returns edited image(s).
    """

    # --- seed handling (preserve your behavior)
    if seed == '':
        seed = DEFAULT_SEED
    else:
        seed = int(seed)
    torch.manual_seed(seed)

    # --- helpers
    def _looks_like_image(obj):
        try:
            import numpy as _np
            from PIL import Image as _PILImage
            return isinstance(obj, (_np.ndarray, _PILImage.Image, bytes, bytearray))
        except Exception:
            return False

    def _extract_image_and_mask(inp):
        # tuple/list (image, mask)
        if isinstance(inp, (list, tuple)) and len(inp) >= 1:
            img_cand = inp[0]
            mask_cand = inp[1] if len(inp) > 1 else None
            return img_cand, mask_cand

        # dict-like payloads (try many key names)
        if isinstance(inp, dict):
            img_keys = ("image", "img", "data", "input", "image_data", "image0")
            mask_keys = ("mask", "mask_image", "maskData", "mask0")
            img_candidate = None
            mask_candidate = None
            for k in img_keys:
                if k in inp and _looks_like_image(inp[k]):
                    img_candidate = inp[k]
                    break
            for k in mask_keys:
                if k in inp and _looks_like_image(inp[k]):
                    mask_candidate = inp[k]
                    break
            if img_candidate is None:
                for v in inp.values():
                    if _looks_like_image(v):
                        img_candidate = v
                        break
            if mask_candidate is None:
                img_like_values = [v for v in inp.values() if _looks_like_image(v)]
                if len(img_like_values) >= 2:
                    mask_candidate = img_like_values[1]
            return img_candidate, mask_candidate

        # direct numpy/PIL/bytes
        try:
            import numpy as _np
            from PIL import Image as _PILImage
            if isinstance(inp, (_np.ndarray, _PILImage.Image, bytes, bytearray)):
                return inp, None
        except Exception:
            pass

        # nothing matched -> debug and return None, None
        print("DEBUG: _extract_image_and_mask got unexpected input type:", type(inp), file=sys.stderr)
        try:
            if isinstance(inp, dict):
                print("DEBUG: dict keys:", list(inp.keys())[:20], file=sys.stderr)
                sample = {k: type(v) for k, v in list(inp.items())[:10]}
                print("DEBUG: sample types:", sample, file=sys.stderr)
            else:
                print("DEBUG: repr(inp)[:300] =", repr(inp)[:300], file=sys.stderr)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return None, None

    # --- Extract and normalize image + mask BEFORE using them
    img_field, mask_field = _extract_image_and_mask(image)

    if img_field is None:
        raise ValueError(
            "Could not find an input image. Check server logs for DEBUG info about the payload shape. "
            "If your UI uses separate components for image and mask, change run() signature accordingly."
        )

    if mask_field is None:
        raise ValueError(
            "No mask provided. Expected combined input like {'image':..., 'mask':...} or a tuple (image, mask)."
        )

    try:
        # Convert to PIL and normalize size
        init_image = to_pil_image(img_field)
        init_image = resize_and_crop(init_image, (EDITOR_IMAGE_SIZE, EDITOR_IMAGE_SIZE))

        # Convert mask to single channel, invert (your original behavior), and resize to match
        mask_image = to_pil_image(mask_field)
        mask_image = mask_image.convert("L")
        mask_image = ImageOps.invert(mask_image)
        mask_image = resize_and_crop(mask_image, init_image.size)

        # --- Optional immunize step (after we have images)
        if immunize:
            immunized_image, _ = immunize_image(
                init_image,
                mask_image,
                pipe_inpaint,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                config=DEFAULT_IMMUNIZATION_CONFIG,
                seed=seed,
            )

        # --- call the inpainting pipeline (now init_image and mask_image are guaranteed)
        image_edited = pipe_inpaint(
            prompt=prompt,
            image=init_image if not immunize else immunized_image,
            mask_image=mask_image,
            height=init_image.size[0],
            width=init_image.size[1],
            eta=1,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # --- postprocess / recover original colors / compose
        image_edited = recover_image(image_edited, init_image, mask_image)

        if immunize:
            return [(immunized_image, 'Immunized Image'), (image_edited, 'Edited After Immunization')]
        return [(image_edited, 'Edited Image (Without Immunization)')]
    finally:
        _cleanup_runtime_memory()

description='''<u>Official</u> demo of our paper: <br>
**Raising the Cost of Malicious AI-Powered Image Editing** <br>
*[Hadi Salman](https://twitter.com/hadisalmanX), [Alaa Khaddaj](https://twitter.com/Alaa_Khaddaj), [Guillaume Leclerc](https://twitter.com/gpoleclerc), [Andrew Ilyas](https://twitter.com/andrew_ilyas), [Aleksander Madry](https://twitter.com/aleks_madry)* <br>
MIT &nbsp;&nbsp;[Paper](https://arxiv.org/abs/2302.06588) 
&nbsp;&nbsp;[Blog post](https://gradientscience.org/photoguard/) 
&nbsp;&nbsp;[![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MadryLab/photoguard)
<br />
Below you can test our (encoder attack) immunization method for making images resistant to manipulation by Stable Diffusion. This immunization process forces the model to perform unrealistic edits. 

**See Section 5 in our paper for a discussion of the intended use cases for (as well as limitations of) this tool.**
<br />
'''

examples_list = [
                    ['./images/hadi_and_trevor.jpg', 'man attending a wedding', '329357', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
                    ['./images/trevor_2.jpg', 'two men in prison', '329357', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
                    ['./images/elon_2.jpg', 'man in a metro station', '214213', GUIDANCE_SCALE, NUM_INFERENCE_STEPS],
                ]


with gr.Blocks() as demo:
    gr.HTML(value="""<h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Interactive Demo: Raising the Cost of Malicious AI-Powered Image Editing </h1><br>
        """)
    gr.Markdown(description)
    with gr.Accordion(label='How to use (step by step):', open=False):
        gr.Markdown('''
            *First, let's edit your image:*        
            + Upload an image (or select from the examples below)
            + Use the brush to mask the parts of the image you want to keep unedited (e.g., faces of people)
            + Add a prompt to guide the edit (see examples below)
            + Play with the seed and click submit until you get a realistic edit that you are happy with (we provided good example seeds for you below)

            *Now, let's immunize your image and try again:*
            + Click on the "Immunize" button, then submit.
            + You will get an immunized version of the image (which should look essentially identical to the original one) as well as its edited version (which should now look rather unrealistic)
        ''')

    with gr.Accordion(label='Example (video):', open=False):
        gr.HTML('''
            <center>
            <iframe width="920" height="600" src="https://www.youtube.com/embed/aTC59Q6ZDNM">
            allow="fullscreen;" frameborder="0">
            </iframe>
            </center>
        '''
        )

    with gr.Row():  
        with gr.Column():
            imgmask = _make_image_mask()
            prompt = gr.Textbox(label='Prompt', placeholder='A photo of a man in a wedding')
            seed = gr.Textbox(label='Seed (Change to get different edits)', placeholder=str(DEFAULT_SEED), visible=True)
            with gr.Accordion("Advanced Options", open=False):
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=25.0, value=GUIDANCE_SCALE, step=0.1)
                num_steps = gr.Slider(label="Number of Inference Steps", minimum=10, maximum=250, value=NUM_INFERENCE_STEPS, step=5)
            immunize = gr.Checkbox(label='Immunize', value=False)
            b1 = gr.Button('Submit')
        with gr.Column():
            #genimages = gr.Gallery(label="Generated images", 
                       #show_label=False, 
                       #elem_id="gallery").style(grid=[1,2], height="auto")
            genimages = gr.Gallery(
                        label="Generated images",
                        elem_id="gallery",
                        columns=2,
                        height="auto",
                        format="webp",
                                    )
            duplicate = gr.HTML("""
                <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
                <br/>
                <a href="https://huggingface.co/spaces/hadisalman/photoguard?duplicate=true">
                <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                <p/>
            """)
            
    b1.click(run, [imgmask, prompt, seed, scale, num_steps, immunize], [genimages])
    examples = gr.Examples(examples=examples_list,inputs = [imgmask, prompt, seed, scale, num_steps, immunize],  outputs=[genimages], cache_examples=False, fn=run)


# demo.launch()
demo.launch(server_name='0.0.0.0', share=False, server_port=7860, inline=False)
