import ort, { InferenceSession } from "onnxruntime-web";

import * as postprocessing from "modules";
import {
	hasRequiredAttributes,
	KeyValuePairs,
	ModelImage,
	runInference,
	Stage,
} from "modules";
import {
	InferenceInput,
	VisionModelTrainerBase,
} from "../base/VisionModelTrainerBase";
import { ClassificationResult } from "./ClassificationResult";
import { ImageClassificationCard } from "./ImageClassificationCard";

export class ClassificationModelTrainer extends VisionModelTrainerBase<ClassificationResult> {
	private classesAttribute = "classes";
	private inputImageSizeAttribute = "input-image-size";
	private normRangeAtribute = "norm-range";

	private classes: string[] = [];
	private inputImageSize: number = 224;
	private normRange: [number, number] = [-1, 1];

	protected override uploadButtonHintText = "";

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

		this.classes = this.getAttribute(this.classesAttribute)!.split(",");
		this.inputImageSize = parseInt(
			this.getAttribute(this.inputImageSizeAttribute)!
		);
		const normRangeStr = this.getAttribute(this.normRangeAtribute)!;
		this.normRange = normRangeStr.split(",").map((n) => parseInt(n)) as [
			number,
			number
		];

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

	protected override async runInference(
		inferenceSession: ort.InferenceSession,
		inferenceInput: InferenceInput<ClassificationResult>[]
	) {
		for (const input of inferenceInput) {
			const { imgElement, id, container } = input;
			this.updateInferenceLoading(container, id, "Running Inference...");
			const outputs = await runInference(
				inferenceSession,
				imgElement,
				this.inputImageSize,
				this.normRange
			);
			const predictionResult = this.modelOutputsToPredictionResult(
				outputs,
				inferenceSession
			);
			await this.updateImageData(id, {
				predictionResult,
			});
			this.updatePredictionResult(container, id, predictionResult);
		}
	}

	private updatePredictionResult(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		predicitonResult: any
	) {
		const labelElement = container.querySelector(
			`#img-label-${imageId}`
		) as HTMLSpanElement;
		labelElement.innerText = predicitonResult.label;
	}

	private updateInferenceLoading(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		message: string
	) {
		const labelElement = container.querySelector(
			`#img-label-${imageId}`
		) as HTMLSpanElement;
		labelElement.innerText = message;
	}

	private modelOutputsToPredictionResult(
		outputs: ort.InferenceSession.OnnxValueMapType,
		session: InferenceSession
	) {
		const output = outputs[session.outputNames[0]];
		var outputSoftmax = postprocessing.softmax(
			Array.prototype.slice.call(output.data)
		);
		var results = postprocessing.classesTopK(
			outputSoftmax,
			Math.min(5, this.classes.length),
			this.classes
		);

		return {
			label: results[0].label,
		};
	}
}
