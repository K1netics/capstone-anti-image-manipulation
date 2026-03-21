from io import BytesIO
import requests
import gradio as gr
import requests
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToPILImage
from utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop
import numpy as np
import sys, traceback

gr.close_all()
topil = ToPILImage()

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "/home/tobi/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-inpainting/snapshots/5f74973cbb64c8568780732c17f43eb269d63a0d",
    local_files_only=True,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_inpaint = pipe_inpaint.to("cuda")

## Good params for editing that we used all over the paper --> decent quality and speed   
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100
DEFAULT_SEED = 1234

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

# --- then inside your run(...) function, replace the line that does:
# init_image = Image.fromarray(image['image'])
# with:

def pgd(X, targets, model, criterion, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()
        pbar.set_description(f"Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

def get_target():
    target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    return target_image

def immunize_fn(init_image, mask_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare inputs on device & half precision
    mask, X = prepare_mask_and_masked_image(init_image, mask_image)
    X = X.half().to(device)
    mask = mask.half().to(device)

    # compute targets WITHOUT building a grad graph (no need to backprop into target encoder)
    with torch.no_grad():
        # ensure preprocess(...) returns a tensor on CPU; move to device and half then encode
        tgt = preprocess(get_target()).half().to(device)
        targets = pipe_inpaint.vae.encode(tgt).latent_dist.mean.detach()

    # Run PGD with mixed precision for forward/backward where possible
    # PGD is expected to perform gradient updates, so we keep gradients enabled.
    # If pgd internally expects X to require_grad, ensure it does.
    # Use autocast to reduce memory for kernel ops.
    adv_X = None
    scaler = None
    try:
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            adv_X = pgd(
                X,
                targets=targets,
                model=pipe_inpaint.vae.encode,
                criterion=torch.nn.MSELoss(),
                clamp_min=-1,
                clamp_max=1,
                eps=0.12,
                step_size=0.01,
                iters=200,
                mask=1 - mask
            )
    finally:
        # best-effort cleanup of temporaries
        try:
            del X, mask, tgt, targets
        except Exception:
            pass

    # postprocess and transfer to PIL
    # ensure adv_X is on CPU for topil if that helper expects CPU tensors
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    # move to CPU and convert if needed by your topil implementation
    adv_image = topil(adv_X[0].detach().cpu()).convert("RGB")
    adv_image = recover_image(adv_image, init_image, mask_image, background=True)

    # synchronize and free cached GPU memory to avoid incremental growth between runs
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return adv_image

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

    # Convert to PIL and normalize size
    init_image = to_pil_image(img_field)
    init_image = resize_and_crop(init_image, (512, 512))

    if mask_field is None:
        raise ValueError(
            "No mask provided. Expected combined input like {'image':..., 'mask':...} or a tuple (image, mask)."
        )

    # Convert mask to single channel, invert (your original behavior), and resize to match
    mask_image = to_pil_image(mask_field)
    mask_image = mask_image.convert("L")
    mask_image = ImageOps.invert(mask_image)
    mask_image = resize_and_crop(mask_image, init_image.size)

    # --- Optional immunize step (after we have images)
    if immunize:
        immunized_image = immunize_fn(init_image, mask_image)

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

    # --- return same structure you had previously
    if immunize:
        return [(immunized_image, 'Immunized Image'), (image_edited, 'Edited After Immunization')]
    else:
        return [(image_edited, 'Edited Image (Without Immunization)')]

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
            imgmask = gr.ImageMask(label='Drawing tool to mask regions you want to keep, e.g. faces')
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
