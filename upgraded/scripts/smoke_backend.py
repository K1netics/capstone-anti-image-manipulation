from __future__ import annotations

import argparse
import json
import mimetypes
import time
from pathlib import Path
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _multipart_encode(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----photoguard-boundary-{int(time.time() * 1000)}"
    lines: list[bytes] = []

    for key, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    for key, (filename, content, content_type) in files.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )

    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


def _get_json(url: str) -> dict:
    with request.urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, *, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> dict:
    body, content_type = _multipart_encode(fields=fields, files=files)
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _poll_json(url: str, *, attempts: int = 180, delay_seconds: float = 2.0) -> dict:
    last_payload: dict | None = None
    for _ in range(attempts):
        try:
            payload = _get_json(url)
        except error.HTTPError as exc:
            if exc.code == 404:
                time.sleep(delay_seconds)
                continue
            raise
        last_payload = payload
        if payload.get("status") in {"completed", "failed"}:
            return payload
        time.sleep(delay_seconds)
    raise TimeoutError(f"Timed out waiting for completion: {last_payload}")


def _read_file(path: Path) -> tuple[str, bytes, str]:
    guessed_type, _ = mimetypes.guess_type(path.name)
    return path.name, path.read_bytes(), guessed_type or "application/octet-stream"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the upgraded PhotoGuard backend.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--image",
        default="/home/tobi/photoguard/test-images/orginal/1-Female_Face.jpg",
    )
    parser.add_argument(
        "--mask",
        default="/home/tobi/photoguard/test-images/masked-auto/Masked-1-Female_Face.png",
    )
    parser.add_argument("--prompt", default="red shirt and sunglasses")
    parser.add_argument("--immunize", action="store_true")
    parser.add_argument("--profile", default="stable_diffusion")
    parser.add_argument("--working-resolution", default="1024")
    parser.add_argument("--guidance-scale", default="7.5")
    parser.add_argument("--num-inference-steps", default="50")
    parser.add_argument("--timeout-polls", type=int, default=180)
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    if not image_path.exists():
        raise SystemExit(f"Missing image: {image_path}")
    if not mask_path.exists():
        raise SystemExit(f"Missing mask: {mask_path}")

    health = _get_json(f"{args.base_url.rstrip('/')}/health")
    print("health:", json.dumps(health, indent=2))

    fields = {
        "prompt": args.prompt,
        "seed": "1234",
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "immunize": "true" if args.immunize else "false",
        "immunization_profile": args.profile,
        "working_resolution": args.working_resolution,
        "output_format": "png",
        "lossless_output": "true",
        "request_id": f"smoke-{int(time.time())}",
    }
    files = {
        "image": _read_file(image_path),
        "mask": _read_file(mask_path),
    }

    start_payload = _post_json(f"{args.base_url.rstrip('/')}/process/start", fields=fields, files=files)
    request_id = start_payload["requestId"]
    print("start:", json.dumps(start_payload, indent=2))

    progress_payload = _poll_json(
        f"{args.base_url.rstrip('/')}/progress/{parse.quote(request_id)}",
        attempts=args.timeout_polls,
    )
    print("progress:", json.dumps(progress_payload, indent=2))

    if progress_payload.get("status") != "completed":
        return 1

    result_payload = _get_json(f"{args.base_url.rstrip('/')}/result/{parse.quote(request_id)}")
    print("result keys:", sorted(result_payload.keys()))
    print("outputs:", len(result_payload.get("outputs", [])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
