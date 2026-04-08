from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_FORMATS_TEXT = ".jpg, .jpeg, .png, .bmp, .tiff, or .webp"


@dataclass
class Feedback:
    """Collects user-friendly status messages for CLI and UI flows."""

    echo: bool = True
    messages: list[str] = field(default_factory=list)

    def _add(self, level: str, msg: str) -> None:
        entry = f"[{level}] {msg}"
        self.messages.append(entry)
        if self.echo:
            print(entry)

    def info(self, msg: str) -> None:
        self._add("INFO", msg)

    def processing(self, msg: str) -> None:
        self._add("PROCESSING", msg)

    def success(self, msg: str) -> None:
        self._add("SUCCESS", msg)

    def warning(self, msg: str) -> None:
        self._add("WARNING", msg)

    def error(self, msg: str) -> None:
        self._add("ERROR", msg)

    def get_status_text(self) -> str:
        return "\n".join(self.messages)


def validate_image_path(path_str: str, fb: Feedback) -> Path:
    path = Path(path_str).expanduser()

    if not path.exists():
        fb.error("File not found. Please check the path and try again.")
        raise FileNotFoundError(str(path))

    if path.suffix.lower() not in ALLOWED_EXTS:
        fb.error(f"Unsupported file format. Please upload {SUPPORTED_FORMATS_TEXT}.")
        raise ValueError(f"Unsupported extension: {path.suffix}")

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        fb.error("This file could not be read as a valid image (it may be corrupted).")
        raise ValueError("Invalid or corrupted image") from exc

    fb.success("Image loaded successfully.")
    return path


def validate_image_bytes(
    data: bytes,
    filename: str | None,
    fb: Feedback,
    *,
    label: str = "Image",
) -> None:
    if not data:
        fb.error(f"{label} upload was empty.")
        raise ValueError("Empty upload")

    suffix = Path(filename or "").suffix.lower()
    if suffix:
        if suffix not in ALLOWED_EXTS:
            fb.error(f"Unsupported {label.lower()} format. Please upload {SUPPORTED_FORMATS_TEXT}.")
            raise ValueError(f"Unsupported extension: {suffix}")
    else:
        fb.warning(f"{label} had no file extension. Validating file contents instead.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        fb.error(f"{label} could not be read as a valid image.")
        raise ValueError("Invalid or corrupted image") from exc

    fb.success(f"{label} validated successfully.")
