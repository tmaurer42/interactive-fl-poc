import { VisionTrainerModalBase } from "components/base";
import {
	createTrainingSession,
	preprocessImagesFromBase64,
	Stage,
} from "modules";
import {
	batchify,
	hasRequiredAttributes,
	shuffleArray,
} from "modules/utils/helpers";
import { Tensor } from "onnxruntime-web";
import { ClassificationResult } from "./ClassificationResult";

type ClassificationTrainerModalProps = {
	classes: string[];
	inputImageSize: number;
	normRange: [number, number];
};

export class ClassificationTrainerModal extends VisionTrainerModalBase<ClassificationTrainerModalProps> {
	private classesAttribute = "classes";
	private inputImageSizeAttribute = "input-image-size";
	private normRangeAtribute = "norm-range";

	constructor() {
		super();
	}

	connectedCallback(): void {
		if (
			!hasRequiredAttributes(this, [
				this.classesAttribute,
				this.inputImageSizeAttribute,
				this.normRangeAtribute,
			])
		) {
			this.innerHTML = "";
			return;
		}

		this.properties.classes = this.getAttribute(
			this.classesAttribute
		)!.split(",");
		this.properties.inputImageSize = parseInt(
			this.getAttribute(this.inputImageSizeAttribute)!
		);
		this.properties.normRange = this.getAttribute(this.normRangeAtribute)!
			.split(",")
			.map((n) => parseInt(n)) as [number, number];

		super.connectedCallback();
	}

	protected async train(
		onTrainStart: () => void,
		updateTrainingProgress: (epoch: number, loss: number) => void,
		sendUpdate: (params: Float32Array) => Promise<void>,
		onTrainEnd: () => void,
		updateProgressMessage: (message: string) => void
	) {
		const {
			classes,
			inputImageSize,
			normRange,
			batchSize,
			nEpochs,
			trainingModelUrl,
			optimizerModelUrl,
			evalModelUrl,
			checkpointUrl,
		} = this.properties;

		onTrainStart();

		const trainIds = await this.repository.getAllIds(
			Stage.ReadyForTraining
		);

		updateProgressMessage("Creating training session...");
		const trainingSession = await createTrainingSession(
			trainingModelUrl,
			optimizerModelUrl,
			evalModelUrl,
			checkpointUrl
		);
		const epochs = Array.from(Array(nEpochs), (_, i) => i + 1);

		updateProgressMessage("Running training process");
		for (let epoch of epochs) {
			shuffleArray(trainIds);
			const batches = batchify(trainIds, batchSize);

			let runningLoss = 0.0;
			let numBatches = 0;
			for (const batch of batches) {
				const trainData =
					await this.repository.getImagesByIds<ClassificationResult>(
						batch
					);
				const images = trainData.map((d) => d!.imageData);
				const labels = trainData.map((img) =>
					classes.findIndex(
						(cls) => cls === img?.predictionResult?.label
					)
				);

				const x = await preprocessImagesFromBase64(
					images,
					inputImageSize,
					normRange
				);
				const y = new Tensor("int64", labels);

				const inputName = trainingSession.trainingInputNames[0];
				const labelsName = trainingSession.trainingInputNames[1];

				const feeds = {
					[inputName]: x,
					[labelsName]: y,
				};

				const result = await trainingSession.runTrainStep(feeds);
				const lossName = trainingSession.trainingOutputNames[0];
				const loss = result[lossName].data;
				runningLoss += loss * (images.length / trainIds.length);

				await trainingSession.runOptimizerStep();
				await trainingSession.lazyResetGrad();

				numBatches += 1;
			}

			const epochLoss = runningLoss / numBatches;
			updateTrainingProgress(epoch, epochLoss);
		}

		const newParams = (await trainingSession.getContiguousParameters(true))
			.data;

		updateProgressMessage("Sending update to server...");
		await sendUpdate(newParams);
		trainIds.forEach((id) => {
			this.repository.updateImageData(id, {
				stage: Stage.Trained,
			});
		});

		updateProgressMessage("All done! You can now close this dialog");
		onTrainEnd();
	}
}
