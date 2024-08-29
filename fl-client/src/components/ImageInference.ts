import { createInferenceSession, runInference } from "modules/inference";
import { imageNetClasses } from "modules/utils/imagenetClasses";
import { classesTopK, softmax } from "modules/utils/postprocessing";

export class ImageInferenceElement extends HTMLElement {
	/*
	 * Since this component does not use a shadow DOM, we need to ensure that
	 * the IDs are unique across all instances of this component.
	 */
	idSuffix = Date.now().toString(36);
	inputId = `inference-input-${this.idSuffix}`;
	submitId = `inference-submit-${this.idSuffix}`;
	imageId = `inference-image-${this.idSuffix}`;
	resultId = `inference-result-${this.idSuffix}`;

	constructor() {
		super();
	}

	connectedCallback() {
		this.render();
		this.addEventListeners();
	}

	render() {
		this.innerHTML = /*html*/ `
			<div class="columns">
				<div class="column is-half">
					<div class="block">
						<label class="label">Select an Image</label>
						<div class="control">
							<input class="input" type="file" name="image" accept="image/*" id="${this.inputId}" autocomplete="off">
						</div>
					</div>
					<div class="block">
						<div class="control">
							<button class="button is-primary" id="${this.submitId}" disabled type="submit">Run Inference!</button>
						</div>
					</div>
					<div class="block">
						<div id="${this.resultId}"></div>
					</div>
				</div>
				<div class="column is-half">
					<figure class="image" id="image-preview">
						<img id="${this.imageId}" src="" alt="">
					</figure>
				</div>
			</div>
        `;
	}

	addEventListeners() {
		this.querySelector(`#${this.inputId}`)!.addEventListener(
			"change",
			this.onInferenceInputChange
		);

		this.querySelector(`#${this.submitId}`)!.addEventListener(
			"click",
			this.onInferenceSubmit
		);
	}

	onInferenceInputChange = (event: Event) => {
		const file = (event.target as HTMLInputElement).files![0];
		const reader = new FileReader();
		reader.onload = (e) => {
			const imagePreview = this.querySelector(
				`#${this.imageId}`
			) as HTMLImageElement;
			imagePreview.src = e.target!.result as string;
			this.querySelector(`#${this.submitId}`)!.removeAttribute(
				"disabled"
			);

			const resultContainer = this.querySelector(`#${this.resultId}`);
			resultContainer!.innerHTML = "";
		};
		reader.readAsDataURL(file);
	};

	onInferenceSubmit = async () => {
		const resultContainer = this.querySelector(`#${this.resultId}`);
		resultContainer!.innerHTML = "Running inference...";

		const imageElement = this.querySelector(
			`#${this.imageId}`
		) as HTMLImageElement;

		const session = await createInferenceSession(
			`/static/models/MobileNet/model.onnx`
		);
		const outputs = await runInference(session, imageElement, 224, [-1, 1]);
		const output = outputs[session.outputNames[0]];
		//Get the softmax of the output data. The softmax transforms values to be between 0 and 1
		var outputSoftmax = softmax(Array.prototype.slice.call(output.data));
		//Get the top 5 results.
		var results = classesTopK(outputSoftmax, 5, imageNetClasses);

		this.displayResult(results);
	};

	displayResult(
		result: Array<{
			label: string;
			probability: number;
		}>
	) {
		const resultContainer = this.querySelector(`#${this.resultId}`);
		resultContainer!.innerHTML = /*html*/ `<h5 class='title is-5'>Top 5 Results</h5>`;

		result.forEach((item, i) => {
			const label = document.createElement("p");
			label.textContent = `${i + 1}: ${
				item.label
			} (${item.probability.toFixed(2)})`;
			resultContainer!.appendChild(label);
		});
	}
}
