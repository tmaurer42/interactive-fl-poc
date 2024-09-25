import ort, { InferenceSession } from "onnxruntime-web";

import { VisionModelTrainerBase } from "./VisionModelTrainerBase";
import { KeyValuePairs, ModelImage, Stage } from "modules/ImageRepository";
import * as postprocessing from "modules/utils/postprocessing";
import { ImageClassificationCard } from "./ImageCard";

type ClassificationResult = {
	label: string;
};

export class ClassificationModelTrainer extends VisionModelTrainerBase<ClassificationResult> {
	private classes = ["AMD", "NO"];

	protected override uploadButtonHintText = `
		Please provide images for the following classes: ${this.classes.join(", ")}`;

	constructor() {
		super();
	}

	private renderLabel(imageId: number, text: string, stage: Stage) {
		const span = document.createElement("span");
		span.id = `img-label-${imageId}`;
		span.innerText = text;

		if (stage === Stage.ReadyForTraining) {
			span.classList.add("has-text-success");
		}

		return span;
	}

	protected override renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<KeyValuePairs>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	) {
		const imageCard = document.createElement(
			"image-card"
		) as ImageClassificationCard;
		imageCard.data = {
			imageId,
			modelImage,
			classes: this.classes,
			onImageLoaded,
			renderLabel: this.renderLabel,
		};

		imageCard.addEventListener("accept-label", (event: any) => {
			const { imageId } = event.detail;
			this.updateImageData(imageId, {
				stage: Stage.ReadyForTraining,
			}).then(() => this.updateProgressDisplay());
		});

		imageCard.addEventListener("change-label", (event: any) => {
			const { imageId, newLabel } = event.detail;
			this.updateImageData(imageId, {
				predictionResult: {
					label: newLabel,
				},
				stage: Stage.ReadyForTraining,
			}).then(() => this.updateProgressDisplay());
		});

		imageCard.addEventListener("delete-image", async (event: any) => {
			const { imageId } = event.detail;
			await this.deleteImage(imageId);
			container.remove();
		});

		container.appendChild(imageCard);
	}

	protected override updatePredictionResult(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		predicitonResult: any
	) {
		const labelElement = container.querySelector(
			`#img-label-${imageId}`
		) as HTMLSpanElement;
		labelElement.innerText = predicitonResult.label;
	}

	protected override updateInferenceLoading(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		message: string
	) {
		const labelElement = container.querySelector(
			`#img-label-${imageId}`
		) as HTMLSpanElement;
		labelElement.innerText = message;
	}

	protected override modelOutputsToPredictionResult(
		outputs: ort.InferenceSession.OnnxValueMapType,
		session: InferenceSession
	) {
		const output = outputs[session.outputNames[0]];
		var outputSoftmax = postprocessing.softmax(
			Array.prototype.slice.call(output.data)
		);
		var results = postprocessing.classesTopK(
			outputSoftmax,
			2,
			this.classes
		);

		return {
			label: results[0].label,
		};
	}
}
