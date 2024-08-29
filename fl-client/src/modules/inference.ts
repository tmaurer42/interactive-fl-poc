import ort, { Tensor } from "onnxruntime-web";
import ortWebGPU from "onnxruntime-web/webgpu";
import { fetchAsArrayBuffer } from "modules/utils/helpers";
import { preprocessImage } from "modules/utils/preprocessing";

export type SupportedModel = "SqueezeNet" | "MobileNet";

export async function createInferenceSession(modelUrl: string) {
	const model = await fetchAsArrayBuffer(modelUrl);
	const supportsWebGPU = Boolean(navigator.gpu);
	const executionProviders = supportsWebGPU ? ["webgpu"] : undefined;

	return supportsWebGPU
		? await ortWebGPU.InferenceSession.create(model, { executionProviders })
		: await ort.InferenceSession.create(model, { executionProviders });
}

export const runInference = async (
	session: ort.InferenceSession,
	imageElement: HTMLImageElement,
	imageTargetSize: number,
	normRange: [number, number]
) => {
	const imageTensor = preprocessImage(
		imageElement,
		imageTargetSize,
		normRange
	);
	const inputName = session.inputNames[0];
	const feeds = { [inputName]: imageTensor };
	const outputData = await session.run(feeds);

	return outputData;
};

export const ensureOnnxRuntime = async () => {
	try {
		var modelUrl = "/static/models/testmodel.onnx";
		// create a new session and load the specific model.
		//
		// the model in this example contains a single MatMul node
		// it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
		// it has 1 output: 'c'(float32, 3x3)
		const session = await createInferenceSession(modelUrl);

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
