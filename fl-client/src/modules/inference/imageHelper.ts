import { Tensor } from "onnxruntime-web";


const createCanvas = (targetSize: number): CanvasRenderingContext2D => {
	const canvas = document.createElement("canvas");
	canvas.width = targetSize;
	canvas.height = targetSize;

	const ctx = canvas.getContext("2d");

	if (!ctx) {
		throw new Error("Could not create canvas 2d context");
	}

	return ctx;
};

const resizeImage = (
	ctx: CanvasRenderingContext2D,
	imgElement: HTMLImageElement,
	targetSize: number
): ImageData => {
	ctx.drawImage(imgElement, 0, 0, targetSize, targetSize);
	return ctx.getImageData(0, 0, targetSize, targetSize);
};

const imageDataToFloat32Array = (
	imageData: ImageData,
	targetSize: number,
	normalizePixelComponent: (value: number) => number
): Float32Array => {
	const data = new Float32Array(targetSize * targetSize * 3);

	for (let i = 0; i < targetSize * targetSize; i++) {
		data[i] = normalizePixelComponent(imageData.data[i * 4]);
		data[i + targetSize * targetSize] = normalizePixelComponent(
			imageData.data[i * 4 + 1]
		);
		data[i + targetSize * targetSize * 2] = normalizePixelComponent(
			imageData.data[i * 4 + 2]
		);
	}

	return data;
};

const getPixelScaler = (low: number, high: number) => (value: number) => {
	if (low >= high) {
        throw new Error("Lower limit must be less than upper limit.");
    }

    const normalizedValue = value / 255;

    return normalizedValue * (high - low) + low;
}

/**
 * Preprocess the image for the model.
 * @param imgElement
 * The image element to preprocess.
 * @param modelName
 * The model name to preprocess the image for.
 * @param targetSize
 * The target size (image height and width) of the image.
 * The default value is 224.
 * @returns
 * The preprocessed image tensor.
 */
const preprocessImage = (
	imgElement: HTMLImageElement,
	targetSize: number = 224,
	scaleRange: [number, number] = [-1, 1]
): Tensor => {
	const ctx = createCanvas(targetSize);
	const imageData = resizeImage(ctx, imgElement, targetSize);
	const data = imageDataToFloat32Array(
		imageData,
		targetSize,
		getPixelScaler(scaleRange[0], scaleRange[1])
	);
	const tensor = new Tensor("float32", data, [1, 3, targetSize, targetSize]);
	
	return tensor
};

export default { preprocessImage };
