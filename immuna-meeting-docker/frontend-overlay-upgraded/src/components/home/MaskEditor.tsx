import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  type PointerEvent as ReactPointerEvent,
} from "react";

export interface MaskEditorHandle {
  clearMask: () => void;
  exportMaskFile: () => Promise<File | null>;
}

interface MaskEditorProps {
  imageUrl: string | null;
  imageWidth: number | null;
  imageHeight: number | null;
  brushSize: number;
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error("Unable to export the mask."));
    }, "image/png");
  });
}

const MaskEditor = forwardRef<MaskEditorHandle, MaskEditorProps>(function MaskEditor(
  { imageUrl, imageWidth, imageHeight, brushSize },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const hasImage = Boolean(imageUrl && imageWidth && imageHeight);

  const clearMask = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
  };

  useImperativeHandle(ref, () => ({
    clearMask,
    async exportMaskFile() {
      const canvas = canvasRef.current;
      if (!canvas) {
        return null;
      }

      const blob = await canvasToBlob(canvas);
      return new File([blob], "mask.png", { type: "image/png" });
    },
  }));

  useEffect(() => {
    clearMask();
  }, [imageUrl]);

  const paint = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }

    const bounds = canvas.getBoundingClientRect();
    const scaleX = canvas.width / bounds.width;
    const scaleY = canvas.height / bounds.height;
    const x = (event.clientX - bounds.left) * scaleX;
    const y = (event.clientY - bounds.top) * scaleY;

    context.strokeStyle = "#ffffff";
    context.fillStyle = "#ffffff";
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = brushSize;
    context.globalAlpha = 1;

    if (!isDrawingRef.current) {
      context.beginPath();
      context.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      context.fill();
      context.beginPath();
      context.moveTo(x, y);
      isDrawingRef.current = true;
      return;
    }

    context.lineTo(x, y);
    context.stroke();
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!imageUrl) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    isDrawingRef.current = false;
    paint(event);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!imageUrl || !isDrawingRef.current) {
      return;
    }

    paint(event);
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!imageUrl) {
      return;
    }

    event.currentTarget.releasePointerCapture(event.pointerId);
    isDrawingRef.current = false;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Protection mask
          </h3>
          <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>
            Paint over the parts of the image you want to keep unchanged. The full frame stays in
            place.
          </p>
        </div>
        <p className="text-xs" style={{ color: "var(--muted-foreground)" }}>
          {hasImage
            ? `Prepared canvas: ${imageWidth} x ${imageHeight}`
            : "Canvas appears after upload"}
        </p>
      </div>

      <div
        className="w-full overflow-hidden rounded-[1.25rem]"
        style={{
          background:
            "linear-gradient(135deg, rgba(248,187,217,0.2), rgba(159,134,192,0.18))",
          border: "1px solid var(--border)",
        }}
      >
        {hasImage ? (
          <div
            className="relative mx-auto"
            style={{
              width: `min(100%, ${imageWidth}px)`,
              aspectRatio: `${imageWidth} / ${imageHeight}`,
            }}
          >
            <img
              src={imageUrl ?? undefined}
              alt="Editor preview"
              className="absolute inset-0 h-full w-full object-contain"
            />
            <canvas
              ref={canvasRef}
              width={imageWidth ?? undefined}
              height={imageHeight ?? undefined}
              className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
              style={{ opacity: 0.58 }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
            />
          </div>
        ) : (
          <div className="flex min-h-[280px] items-center justify-center p-8 text-center">
            <p className="text-sm leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
              Upload an image to start painting the protected regions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
});

export default MaskEditor;
