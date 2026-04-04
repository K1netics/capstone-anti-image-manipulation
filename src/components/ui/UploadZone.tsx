import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  preview: string | null;
}

const ACCEPTED = ["image/png", "image/jpeg", "image/gif", "image/webp"];

export default function UploadZone({ onFileSelect, preview }: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (ACCEPTED.includes(file.type)) {
        onFileSelect(file);
      }
    },
    [onFileSelect],
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
        accept={ACCEPTED.join(",")}
        className="hidden"
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
              PNG, JPG, GIF, or WEBP up to 10MB
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
