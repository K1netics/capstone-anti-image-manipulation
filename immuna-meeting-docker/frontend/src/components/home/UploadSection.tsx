import { useState } from "react";
import Card from "../ui/Card";
import UploadZone from "../ui/UploadZone";
import Button from "../ui/Button";

interface UploadSectionProps {
  onGenerate: (file: File) => Promise<void>;
  isLoading: boolean;
}

export default function UploadSection({ onGenerate, isLoading }: UploadSectionProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileSelect = (f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  return (
    <Card>
      <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--foreground)" }}>
        Upload Image
      </h2>
      <UploadZone onFileSelect={handleFileSelect} preview={preview} />
      <Button fullWidth onClick={async () => { if (file) await onGenerate(file); }} disabled={!file || isLoading} className="mt-4">
        {isLoading ? "Processing…" : "Upload & Generate"}
      </Button>
    </Card>
  );
}
 
