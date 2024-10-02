import ort, { Tensor } from "onnxruntime-web/training";
import { fetchAsUint8Array, getFirstMatchingProperty } from "./utils";

export type ImageLabelPair = {
	image: string;
	label: string;
};

export type TrainStepResult = {
	loss: number;
};

export function* batchify<T>(array: T[], batchSize: number) {
	const numBatches = Math.ceil(array.length / batchSize);

	for (let i = 0; i < numBatches; i++) {
		yield array.slice(i * batchSize, i * batchSize + batchSize);
	}
}

function getLoss(trainOutput: object): number {
	const lossTensor = getFirstMatchingProperty(trainOutput, "onnx::loss::");
	const loss = parseFloat(lossTensor.data);

	return loss;
}

export async function runTrainStep(
	session: ort.TrainingSession,
	x: Tensor,
	y: Tensor
): Promise<TrainStepResult> {
	const feeds = {
		input: x,
		labels: y,
	};
	const trainResult = await session.runTrainStep(feeds);
	await session.runOptimizerStep();
	await session.lazyResetGrad();

	const loss = getLoss(trainResult);

	return {
		loss,
	};
}

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

	const session = await ort.TrainingSession.create(createOptions);

	return session;
}
