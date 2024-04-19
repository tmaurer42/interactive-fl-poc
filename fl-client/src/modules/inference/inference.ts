import { InferenceSession, Tensor } from "onnxruntime-web";

const fetchModel = async (modelUrl: string) => {
	const response = await fetch(modelUrl);
	return await response.arrayBuffer();
};

const fetchClasses = async (classesUrl: string) => {
	// result is a json file with the classes
	const response = await fetch(classesUrl);
	return (await response.json()) as string[];
};

const preprocessImageSqueezeNet = (
	imgElement: HTMLImageElement,
	targetSize: number = 224
): Tensor => {
	const canvas = document.createElement("canvas");
	canvas.width = targetSize;
	canvas.height = targetSize;

	const ctx = canvas.getContext("2d");

	if (!ctx) {
		throw new Error("Could not create canvas 2d context");
	}

	ctx.drawImage(imgElement, 0, 0, targetSize, targetSize);

	const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
	const data = new Float32Array(targetSize * targetSize * 3);

	for (let i = 0; i < targetSize * targetSize; i++) {
		data[i] = imageData.data[i * 4] / 255;
		data[i + targetSize * targetSize] = imageData.data[i * 4 + 1] / 255;
		data[i + targetSize * targetSize * 2] = imageData.data[i * 4 + 2] / 255;
	}

	return new Tensor("float32", data, [1, 3, targetSize, targetSize]);
};

const preprocessImageMobileNet = (
	imgElement: HTMLImageElement,
	targetSize: number = 224
): Tensor => {
	const canvas = document.createElement("canvas");
	canvas.width = targetSize;
	canvas.height = targetSize;

	const ctx = canvas.getContext("2d");

	if (!ctx) {
		throw new Error("Could not create canvas 2d context");
	}

	ctx.drawImage(imgElement, 0, 0, targetSize, targetSize);

	const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
	const data = new Float32Array(targetSize * targetSize * 3);

	for (let i = 0; i < targetSize * targetSize; i++) {
		data[i] = imageData.data[i * 4] / 127.5 - 1;
		data[i + targetSize * targetSize] =
			imageData.data[i * 4 + 1] / 127.5 - 1;
		data[i + targetSize * targetSize * 2] =
			imageData.data[i * 4 + 2] / 127.5 - 1;
	}

	return new Tensor("float32", data, [1, 3, targetSize, targetSize]);
};

type SupportedModel = "SqueezeNet" | "MobileNet";

function preprocessImage(
	imgElement: HTMLImageElement,
	modelName: SupportedModel,
	targetSize: number = 224
): Tensor {
	switch (modelName) {
		case "SqueezeNet":
			return preprocessImageSqueezeNet(imgElement, targetSize);
		case "MobileNet":
			return preprocessImageMobileNet(imgElement, targetSize);
		default:
			throw new Error(`Model ${modelName} not supported`);
	}
}

function getExecutionProviders() {
	// check if browser supports webgpu
	if (navigator.gpu) {
		return ["webgpu"];
	}

	return undefined;
}

export const runInference = async (
	imageElement: HTMLImageElement,
	modelName: "SqueezeNet" | "MobileNet" = "SqueezeNet"
) => {
	const model = await fetchModel(`/static/models/${modelName}/model.onnx`);
	const classes = await fetchClasses("/static/classes/imagenet.json");

	const imageTensor = preprocessImage(imageElement, modelName);
	console.log("Image processed");

	const executionProviders = getExecutionProviders();
	const session = await InferenceSession.create(model, {
		executionProviders,
	});
	const inputName = session.inputNames[0];
	const feeds = { [inputName]: imageTensor };
	const outputData = await session.run(feeds);

	// Get output results with the output name from the model export.
	const output = outputData[session.outputNames[0]];
	//Get the softmax of the output data. The softmax transforms values to be between 0 and 1
	var outputSoftmax = softmax(Array.prototype.slice.call(output.data));
	//Get the top 5 results.
	var results = classesTopK(outputSoftmax, 5, classes);

	console.log("results: ", results);

	return results;
};

//Get the top K classes from the output data.
function classesTopK(
	outputData: number[],
	topK: number,
	classes: string[]
): any {
	//Create an array of indices [0, 1, 2, ..., 999].
	var resultArray = Array.from(Array(outputData.length).keys());
	//Sort the indices based on the output data.
	resultArray.sort((a, b) => outputData[b] - outputData[a]);
	//Get the top K indices.
	return resultArray.slice(0, topK).map((i) => {
		return { label: classes[i], probability: outputData[i] };
	});
}

//The softmax transforms values to be between 0 and 1
function softmax(resultArray: number[]) {
	// Get the largest value in the array.
	const largestNumber = Math.max(...resultArray);
	// Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
	const sumOfExp = resultArray
		.map((resultItem) => Math.exp(resultItem - largestNumber))
		.reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
	//Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
	return resultArray.map((resultValue) => {
		return Math.exp(resultValue - largestNumber) / sumOfExp;
	});
}

export const ensureOnnxRuntime = async () => {
	try {
		var model = await fetchModel("/static/models/testmodel.onnx");
		// create a new session and load the specific model.
		//
		// the model in this example contains a single MatMul node
		// it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
		// it has 1 output: 'c'(float32, 3x3)
		const session = await InferenceSession.create(model);

		// prepare inputs. a tensor need its corresponding TypedArray as data
		const dataA = Float32Array.from([
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		]);
		const dataB = Float32Array.from([
			10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
		]);
		const tensorA = new Tensor("float32", dataA, [3, 4]);
		const tensorB = new Tensor("float32", dataB, [4, 3]);

		// prepare feeds. use model input names as keys.
		const feeds = { a: tensorA, b: tensorB };

		// feed inputs and run
		const results = await session.run(feeds);

		// read from results
		const dataC = results.c.data;
		console.log(
			`Test inference session successful! Data of result tensor 'c': ${dataC}`
		);
	} catch (e) {
		console.log(`Failed to inference test ONNX model: ${e}.`);
	}
};
