import ort from "onnxruntime-web/training";
import { fetchAsUint8Array } from "./utils";

/**
 *
 * @param modelUrl URL to the inference model file.
 * @returns A promise that resolves with the inferecence session.
 */
/**
 * Creates a training session.
 * @param trainingModelUrl URL to the training model file.
 * @param optimizerModelUrl URL to the optimizer model file.
 * @param evalModelUrl URL to the eval model file.
 * @param checkpointUrl URL to the checkpoint file.
 * @returns A promise that resolves with the training session.
 */
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
