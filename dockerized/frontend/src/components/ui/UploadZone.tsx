import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  onError?: (message: string) => void;
  preview: string | null;
}

const ACCEPTED_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
]);
const FILE_INPUT_ACCEPT = "image/*,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tif,.tiff";

function hasAcceptedExtension(fileName: string) {
  const lowerName = fileName.toLowerCase();
  return Array.from(ACCEPTED_EXTENSIONS).some((extension) => lowerName.endsWith(extension));
}

export default function UploadZone({ onFileSelect, onError, preview }: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const looksLikeImage = file.type.startsWith("image/") || hasAcceptedExtension(file.name);
      if (looksLikeImage) {
        onFileSelect(file);
        if (inputRef.current) {
          inputRef.current.value = "";
        }
        return;
      }

      onError?.("That file does not look like a supported image. Try PNG, JPG, WebP, BMP, or TIFF.");
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    },
    [onError, onFileSelect],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  return (
    <div
      className="relative rounded-[1.25rem] border-2 border-dashed flex flex-col items-center justify-center min-h-[220px] cursor-pointer transition-colors duration-200 overflow-hidden"
      style={{
        borderColor: dragOver ? "var(--accent)" : "var(--border)",
        backgroundColor: dragOver ? "rgba(159,134,192,0.08)" : "var(--muted)",
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={FILE_INPUT_ACCEPT}
        className="hidden"
        onClick={(event) => {
          event.currentTarget.value = "";
        }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            handleFile(file);
          }
        }}
      />

      {preview ? (
        <img
          src={preview}
          alt="Selected preview"
          className="max-h-[240px] max-w-full rounded-xl object-contain p-3"
        />
      ) : (
        <div className="flex flex-col items-center gap-3 p-8 text-center select-none">
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center"
            style={{ backgroundColor: "var(--secondary)" }}
          >
            <Upload className="w-5 h-5" style={{ color: "var(--accent)" }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: "var(--foreground)" }}>
              Drag and drop an image here, or <span style={{ color: "var(--accent)" }}>click to select</span>
            </p>
            <p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>
              PNG, JPG, GIF, WEBP, BMP, or TIFF
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
