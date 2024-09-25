import { KeyValuePairs, ModelImage, Stage } from "modules/ImageRepository";

type ImageClassificationCardProps = {
	imageId: number;
	modelImage: ModelImage<KeyValuePairs>;
	classes: string[];
	onImageLoaded: (imgElement: HTMLImageElement) => void;
	renderLabel: (
		imageId: number,
		text: string,
		stage: Stage
	) => HTMLSpanElement;
};

export class ImageClassificationCard extends HTMLElement {
	imageId?: number;
	modelImage?: ModelImage<KeyValuePairs>;
	classes?: string[];
	onImageLoaded?: (imgElement: HTMLImageElement) => void;
	renderLabel?: (
		imageId: number,
		text: string,
		stage: Stage
	) => HTMLSpanElement;

	constructor() {
		super();
	}

	connectedCallback() {
		this.render();
		this.bindImageEvents();
	}

	render() {
		if (
			!(
				this.imageId &&
				this.modelImage &&
				this.classes &&
				this.onImageLoaded &&
				this.renderLabel
			)
		) {
			this.innerHTML = "";
			return;
		}

		const label = this.modelImage.predictionResult?.label ?? "";
		const imgElement = document.createElement("img");
		imgElement.style.maxHeight = "100%";
		imgElement.id = `img-${this.imageId}`;
		imgElement.onload = () => this.onImageLoaded?.(imgElement);
		imgElement.src = this.modelImage.imageData;

		const imageHeight = 10;

		this.innerHTML = `
			<div class="card" id="image-card-${this.imageId}">
				<div>
					<button 
						id="btn-delete-img-${this.imageId}" 
						class="card-header-icon has-text-grey-dark" 
						title="Delete image"
						style="float:right;"
					>
						&#10005;
					</button>
				</div>
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
							${this.renderLabel(this.imageId, label, this.modelImage.stage).outerHTML}
						</strong>
					</label>
					<a class="card-footer-item">
						<div class="buttons are-small">
							<button 
								id="btn-accept-img-label-${this.imageId}" 
								class="button is-success"
								title="Confirm label"
								${this.modelImage.stage === Stage.ReadyForTraining ? "disabled" : ""}
							>
								&#10004;
							</button>
							<button 
								id="btn-change-img-label-${this.imageId}" 
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
	}

	bindImageEvents() {
		const acceptLabelBtn = this.querySelector(
			`#btn-accept-img-label-${this.imageId}`
		) as HTMLButtonElement;
		const changeLabelBtn = this.querySelector(
			`#btn-change-img-label-${this.imageId}`
		) as HTMLButtonElement;
		const deleteImgButton = this.querySelector(
			`#btn-delete-img-${this.imageId}`
		) as HTMLButtonElement;

		acceptLabelBtn.addEventListener("click", () => {
			this.dispatchEvent(
				new CustomEvent("accept-label", {
					detail: { imageId: this.imageId },
				})
			);
			const labelElement = this.querySelector(
				`#img-label-${this.imageId}`
			) as HTMLSpanElement;
			acceptLabelBtn.disabled = true;
			changeLabelBtn.disabled = false;
			labelElement.classList.add("has-text-success");
		});

		changeLabelBtn.addEventListener("click", () => {
			const labelElement = document.querySelector(
				`#img-label-${this.imageId}`
			) as HTMLSpanElement;
			acceptLabelBtn.disabled = true;
			changeLabelBtn.disabled = true;

			const dropdownWrapper = document.createElement("div");
			dropdownWrapper.classList.add("select");
			dropdownWrapper.classList.add("is-small");
			const dropdown = document.createElement("select");
			dropdown.id = `dropdown-label-${this.imageId}`;
			this.classes?.forEach((label) => {
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
				this.dispatchEvent(
					new CustomEvent("change-label", {
						detail: {
							imageId: this.imageId,
							newLabel,
						},
					})
				);
				const newLabelElement = this.renderLabel?.(
					this.imageId!,
					newLabel,
					Stage.ReadyForTraining
				);
				dropdownWrapper.replaceWith(newLabelElement!);

				acceptLabelBtn.disabled = true;
				changeLabelBtn.disabled = false;
			};
		});

		deleteImgButton.addEventListener("click", () => {
			this.dispatchEvent(
				new CustomEvent("delete-image", {
					detail: { imageId: this.imageId },
				})
			);
		});
	}

	set data({
		imageId,
		modelImage,
		classes,
		onImageLoaded,
		renderLabel,
	}: ImageClassificationCardProps) {
		this.imageId = imageId;
		this.modelImage = modelImage;
		this.classes = classes;
		this.onImageLoaded = onImageLoaded;
		this.renderLabel = renderLabel;
		this.render();
	}
}
