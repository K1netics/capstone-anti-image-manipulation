from __future__ import annotations
import argparse
from pathlib import Path

from feedback import Feedback, validate_image_path


def mock_encode_image(input_path: Path) -> Path:
    """
    Placeholder encoder.
    
    """
    output_path = input_path.with_name(f"protected_{input_path.name}")
    output_path.write_bytes(input_path.read_bytes())  # copy for demo
    return output_path


def process_image_for_ui(image_path: str):
    """
    Simulates how the backend would behave for a UI 
    Returns:
        output_path_or_none, status_text
    """
    fb = Feedback(echo=True)

    fb.info("Starting image protection tool.")

    try:
        img_path = validate_image_path(image_path, fb)

        fb.processing("Applying image protection...")
        out_path = mock_encode_image(img_path)

        fb.success(f"Protection successfully applied. Saved as: {out_path.name}")
        return str(out_path), fb.get_status_text()

    except Exception as e:
        fb.info(f"Debug detail: {type(e).__name__}")
        return None, fb.get_status_text()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capstone: user feedback + confirmation messages demo")
    parser.add_argument("image_path", help="Path to the image file to protect")
    args = parser.parse_args()

    out_path, status_text = process_image_for_ui(args.image_path)

    print("\n--- UI STATUS OUTPUT ---")
    print(status_text)

    if out_path:
        print(f"\nOutput file: {out_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
