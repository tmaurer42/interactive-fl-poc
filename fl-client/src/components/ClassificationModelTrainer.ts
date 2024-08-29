import ort, { InferenceSession } from "onnxruntime-web";

import { VisionModelTrainerBase } from "./VisionModelTrainerBase";
import { KeyValuePairs, ModelImage } from "modules/ImageRepository";
import * as postprocessing from "modules/utils/postprocessing";
import { imageNetClasses } from "modules/utils/imagenetClasses";

type ClassificationResult = {
	label: string;
};

export class ClassificationModelTrainer extends VisionModelTrainerBase<ClassificationResult> {
	private classes = imageNetClasses;

	constructor() {
		super();
	}

	protected override renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<KeyValuePairs>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	) {
		const labelElement = document.createElement("label");
		labelElement.id = `img-label-${imageId}`;

		if (modelImage.predictionResult?.label) {
			labelElement.innerHTML = modelImage.predictionResult.label;
		}

		const imgElement = document.createElement("img");
		imgElement.id = `img-${imageId}`;
		imgElement.onload = () => onImageLoaded(imgElement);
		imgElement.src = modelImage.imageData;
		container.appendChild(imgElement);
		container.appendChild(labelElement);
	}

	protected override updatePredictionResult(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		predicitonResult: any
	) {
		const labelElement = container.querySelector(
			`label`
		) as HTMLLabelElement;
		labelElement.innerHTML = predicitonResult.label;
	}

	protected override updateInferenceLoading(
		container: Pick<HTMLDivElement, "querySelector">,
		message: string
	) {
		const labelElement = container.querySelector(
			`label`
		) as HTMLLabelElement;
		labelElement.innerHTML = message;
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
			5,
			this.classes
		);

		return {
			label: results[0].label,
		};
	}
}
