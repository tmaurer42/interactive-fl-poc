import ort, { Tensor } from "onnxruntime-web/training";
import { fetchAsUint8Array, preprocessImage } from "./utils";

export async function createTrainingSession(
	trainingModelUrl: string,
	optimizerModelUrl: string,
	evalModelUrl: string,
	checkpointUrl: string
): Promise<ort.TrainingSession> {
	console.log("Attempting to load training session...");

	const trainModel = await fetchAsUint8Array(trainingModelUrl);
	const optimizerModel = await fetchAsUint8Array(optimizerModelUrl);
	const evalModel = await fetchAsUint8Array(evalModelUrl);
	const checkpointState = await fetchAsUint8Array(checkpointUrl);

	const createOptions: ort.TrainingSessionCreateOptions = {
		trainModel,
		evalModel,
		optimizerModel,
		checkpointState,
	};

	const session = await ort.TrainingSession.create(createOptions, {
		extra: {
			optimizer_config: {
				learning_rate: 1.0,
			},
		},
	});
	const img = document.querySelector("img")!;
	const tensor = preprocessImage(img, 224, [-1, 1]);
	const feeds = {
		input: tensor,
		labels: new ort.Tensor("int64", [0], [1]),
	};
	const foo = await session.runTrainStep(feeds);
	console.log(foo);
	console.log("Running optimizer step");
	await session.runOptimizerStep({});
	console.log("Resetting gradients");
	await session.lazyResetGrad();
	console.log("Training finished");
	const params = await session.getContiguousParameters(true);

	console.log(params);

	return session;
}
