import ort, { InferenceSession } from "onnxruntime-web";

import { VisionModelTrainerBase } from "../base/VisionModelTrainerBase";
import { KeyValuePairs, ModelImage, Stage } from "modules/ImageRepository";
import * as postprocessing from "modules/utils/postprocessing";
import { ImageClassificationCard } from "./ImageCard";
import { ClassificationResult } from "./ClassificationResult";

export class ClassificationModelTrainer extends VisionModelTrainerBase<ClassificationResult> {
	private classesAttribute = "classes";
	private classes: string[] = [];

	protected override uploadButtonHintText = "";

	constructor() {
		super();
	}

	connectedCallback(): void {
		if (!this.hasAttribute(this.classesAttribute)) {
			this.innerHTML = "";
			return;
		}

		this.classes = this.getAttribute(this.classesAttribute)!.split(",");

		this.uploadButtonHintText = `
			Please provide images for the following classes: ${this.classes.join(", ")}`;
		super.connectedCallback();
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
			this.deleteImage(imageId).then(() => this.updateProgressDisplay());
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
