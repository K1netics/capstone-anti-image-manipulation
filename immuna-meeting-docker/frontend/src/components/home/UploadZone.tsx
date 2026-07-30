import { useCallback, useRef, useState } from "react";
import { Upload, AlertCircle, CheckCircle2 } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  preview: string | null;
}

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/gif"];
const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export default function UploadZone({ onFileSelect, preview }: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Invalid file type. Please upload a PNG, JPG, or GIF.");
        return;
      }
      if (file.size > MAX_SIZE_BYTES) {
        setError("File is too large. Maximum size is 10MB.");
        return;
      }
      setFileName(file.name);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndSelect(file);
    },
    [validateAndSelect]
  );

  return (
    <div className="flex flex-col gap-2 h-full">
      <div
        className="relative rounded-xl border-2 border-dashed flex flex-col items-center justify-center flex-1 min-h-[200px] cursor-pointer transition-all duration-200"
        style={{
          borderColor: error ? "#d4183d" : dragOver ? "var(--accent)" : "var(--border)",
          backgroundColor: error ? "rgba(212,24,61,0.04)" : dragOver ? "rgba(159,134,192,0.06)" : "var(--muted)",
        }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) validateAndSelect(file);
            e.target.value = "";
          }}
        />
        {preview ? (
          <div className="flex flex-col items-center gap-2 p-4">
            <img src={preview} alt="Selected preview" className="max-h-36 rounded-lg object-contain" />
            {fileName && (
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" style={{ color: "#22c55e" }} />
                <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>{fileName}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 p-8 text-center select-none">
            <div className="w-12 h-12 rounded-full flex items-center justify-center" style={{ backgroundColor: "var(--secondary)" }}>
              <Upload className="w-5 h-5" style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--foreground)" }}>
                Drag and drop an image here, or <span style={{ color: "var(--accent)" }}>click to select</span>
              </p>
              <p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>PNG, JPG, or GIF up to 10MB</p>
            </div>
          </div>
        )}
      </div>
      {error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs" style={{ backgroundColor: "rgba(212,24,61,0.08)", color: "#d4183d" }}>
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}