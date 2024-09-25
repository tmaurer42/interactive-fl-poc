import ort, { InferenceSession } from "onnxruntime-web";

import { VisionModelTrainerBase } from "./VisionModelTrainerBase";
import { KeyValuePairs, ModelImage, Stage } from "modules/ImageRepository";
import * as postprocessing from "modules/utils/postprocessing";

type ClassificationResult = {
	label: string;
};

export class ClassificationModelTrainer extends VisionModelTrainerBase<ClassificationResult> {
	private classes = ["AMD", "NO"];

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
		const labelElement = document.createElement("label");
		labelElement.id = `img-label-${imageId}`;

		if (modelImage.predictionResult?.label) {
			labelElement.innerHTML = modelImage.predictionResult.label;
		}

		const label = modelImage.predictionResult?.label ?? "";
		const imgContainer = document.createElement("div");
		imgContainer.classList.add();
		const imgElement = document.createElement("img");
		imgElement.style.maxHeight = "100%";
		imgElement.id = `img-${imageId}`;
		imgElement.onload = () => onImageLoaded(imgElement);
		imgElement.src = modelImage.imageData;

		const imageHeight = 13;

		container.innerHTML = `
			<div class="card">
				<div class="card-content">
    				<div 
						class="content is-flex is-justify-content-center is-align-items-center"
						style="height:${imageHeight}em;"
					>
						${imgElement.outerHTML}
					</div>
				</div>
				<footer class="card-footer">
						<label class="card-footer-item">
							<strong>
								${this.renderLabel(imageId, label, modelImage.stage).outerHTML}
							</strong>
						</label>
						<a class="card-footer-item">
							<div class="buttons">
								<button 
									id="btn-accept-img-label-${imageId}" 
									class="button is-success"
									title="Confirm label"
								>
									&#10004;
								</button>
								<button 
									id="btn-change-img-label-${imageId}" 
									class="button is-warning"
									title="Change label"
								>
									&#9998;
								</button>
							</div>
						</a>
				</footer>
			</div>
		`;

		this.bindImageEvents(imageId);
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
			2,
			this.classes
		);

		return {
			label: results[0].label,
		};
	}

	private bindImageEvents(imageId: number): void {
		const acceptLabelBtn = document.querySelector(
			`#btn-accept-img-label-${imageId}`
		) as HTMLButtonElement;

		const changeLabelBtn = document.querySelector(
			`#btn-change-img-label-${imageId}`
		) as HTMLButtonElement;

		const possibleLabels = this.classes;

		acceptLabelBtn.onclick = () => {
			const labelElement = document.querySelector(
				`#img-label-${imageId}`
			) as HTMLSpanElement;

			acceptLabelBtn.disabled = true;
			changeLabelBtn.disabled = false;

			this.updateImageData(imageId, { stage: Stage.ReadyForTraining });
			this.incrementProgress();
			labelElement.classList.add("has-text-success");
		};

		changeLabelBtn.onclick = () => {
			const labelElement = document.querySelector(
				`#img-label-${imageId}`
			) as HTMLSpanElement;

			acceptLabelBtn.disabled = true;
			changeLabelBtn.disabled = true;

			const dropdownWrapper = document.createElement("div");
			dropdownWrapper.classList.add("select");
			const dropdown = document.createElement("select");
			dropdown.id = `dropdown-label-${imageId}`;
			possibleLabels.forEach((label) => {
				const option = document.createElement("option");
				option.value = label;
				option.textContent = label;
				if (label === labelElement.innerText) {
					option.selected = true;
				}
				dropdown.appendChild(option);
			});
			dropdownWrapper.appendChild(dropdown);

			labelElement.replaceWith(dropdownWrapper);

			dropdown.onchange = () => {
				const newLabel = dropdown.value;
				this.updateImageData(imageId, {
					predictionResult: {
						label: newLabel,
					},
					stage: Stage.ReadyForTraining,
				});
				this.incrementProgress();

				const newLabelElement = this.renderLabel(
					imageId,
					newLabel,
					Stage.ReadyForTraining
				);
				dropdownWrapper.replaceWith(newLabelElement);

				acceptLabelBtn.disabled = true;
				changeLabelBtn.disabled = false;
			};
		};
	}
}
