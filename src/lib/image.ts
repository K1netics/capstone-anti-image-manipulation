const MAX_EDITOR_DIMENSION = 512;
const DIMENSION_MULTIPLE = 32;

export interface EditorImageDimensions {
  width: number;
  height: number;
}

function snapDimension(value: number) {
  const safeValue = Math.max(DIMENSION_MULTIPLE, Math.floor(value));
  return Math.max(
    DIMENSION_MULTIPLE,
    Math.floor(safeValue / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE,
  );
}

function calculateEditorDimensions(width: number, height: number): EditorImageDimensions {
  if (width >= height) {
    const targetWidth = snapDimension(Math.min(width, MAX_EDITOR_DIMENSION));
    const scale = targetWidth / width;
    return {
      width: targetWidth,
      height: snapDimension(height * scale),
    };
  }

  const targetHeight = snapDimension(Math.min(height, MAX_EDITOR_DIMENSION));
  const scale = targetHeight / height;
  return {
    width: snapDimension(width * scale),
    height: targetHeight,
  };
}

function loadImage(sourceUrl: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to load the selected image."));
    image.src = sourceUrl;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error("Unable to export image data."));
    }, type, quality);
  });
}

export async function createEditorImage(file: File) {
  const sourceUrl = URL.createObjectURL(file);

  try {
    const image = await loadImage(sourceUrl);
    const dimensions = calculateEditorDimensions(image.width, image.height);
    const canvas = document.createElement("canvas");
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas is unavailable in this browser.");
    }

    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.clearRect(0, 0, dimensions.width, dimensions.height);
    context.drawImage(image, 0, 0, dimensions.width, dimensions.height);

    const blob = await canvasToBlob(canvas, "image/png");
    const preparedFile = new File([blob], "editor-image.png", { type: "image/png" });
    const previewUrl = URL.createObjectURL(blob);

    return {
      preparedFile,
      previewUrl,
      dimensions,
    };
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
}

export const maxEditorDimension = MAX_EDITOR_DIMENSION;
