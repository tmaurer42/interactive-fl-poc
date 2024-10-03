import { Tensor } from "onnxruntime-web";

/**
 * Preprocess the image for the model.
 * @param imgElement The image element to preprocess.
 * @param modelName The model name to preprocess the image for.
 * @param targetSize The target size (image height and width) of the image.
 * The default value is 224.
 * @param normRange The normalization range for pixel values.
 * The default value is [-1, 1].
 * @returns The preprocessed image tensor.
 */
export const preprocessImage = (
	imgElement: HTMLImageElement,
	targetSize: number = 224,
	normRange: [number, number] = [-1, 1]
): Tensor => {
	const ctx = createCanvas(targetSize);
	const imageData = resizeImage(ctx, imgElement, targetSize);
	const data = imageDataToFloat32Array(
		imageData,
		targetSize,
		getPixelScaler(normRange[0], normRange[1])
	);
	const tensor = new Tensor("float32", data, [1, 3, targetSize, targetSize]);

	return tensor;
};

/**
 * Preprocess the image from a base64 string.
 * @param base64String The base64 string of the image.
 * @param targetSize The target size (image height and width) of the image.
 * The default value is 224.
 * @param normRange The normalization range for pixel values.
 * The default value is [-1, 1].
 * @returns A promise that resolves to the preprocessed image tensor.
 */
export const preprocessImageFromBase64 = async (
	base64String: string,
	targetSize: number = 224,
	normRange: [number, number] = [-1, 1]
): Promise<Tensor> => {
	const imgElement = await loadImageFromBase64(base64String);

	return preprocessImage(imgElement, targetSize, normRange);
};

/**
 * Preprocess an array of base64 images into a single tensor.
 * @param base64Images An array of base64 image strings.
 * @param targetSize The target size (image height and width) of the images.
 * The default value is 224.
 * @param normRange The normalization range for pixel values.
 * The default value is [-1, 1].
 * @returns A promise that resolves to a single tensor containing all preprocessed images.
 */
export const preprocessImagesFromBase64 = async (
	base64Images: string[],
	targetSize: number = 224,
	normRange: [number, number] = [-1, 1]
): Promise<Tensor> => {
	const imageTensors = await Promise.all(
		base64Images.map(async (base64Image) => {
			const imgElement = await loadImageFromBase64(base64Image);
			const ctx = createCanvas(targetSize);
			const imageData = resizeImage(ctx, imgElement, targetSize);
			const data = imageDataToFloat32Array(
				imageData,
				targetSize,
				getPixelScaler(normRange[0], normRange[1])
			);

			return data;
		})
	);

	const totalImages = base64Images.length;
	const combinedData = new Float32Array(
		totalImages * 3 * targetSize * targetSize
	);

	for (let i = 0; i < totalImages; i++) {
		const offset = i * 3 * targetSize * targetSize;
		combinedData.set(imageTensors[i], offset);
	}

	const combinedTensor = new Tensor("float32", combinedData, [
		totalImages,
		3,
		targetSize,
		targetSize,
	]);

	return combinedTensor;
};

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
	scalePixel: (value: number) => number
): Float32Array => {
	const data = new Float32Array(targetSize * targetSize * 3);

	for (let i = 0; i < targetSize * targetSize; i++) {
		data[i] = scalePixel(imageData.data[i * 4]);
		data[i + targetSize * targetSize] = scalePixel(
			imageData.data[i * 4 + 1]
		);
		data[i + targetSize * targetSize * 2] = scalePixel(
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
};

const loadImageFromBase64 = (
	base64String: string
): Promise<HTMLImageElement> => {
	return new Promise((resolve) => {
		const img = new Image();
		img.src = base64String;
		img.onload = () => resolve(img);
	});
};
