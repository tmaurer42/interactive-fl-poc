import { runInference } from "modules";
import {
	ImageRepository,
	ModelImage,
	Repository,
	Stage,
} from "modules/repository";

type ClassificationResult = {
	label: string;
};

export class ImageClassificationComponent extends HTMLElement {
	private repository: ImageRepository;
	private imageKey?: number;
	private imageElementId?: string;
	private labelElementId?: string;
	private modelImage?: ModelImage<ClassificationResult>;

	private imageKeyAttribute = "image-key";

	constructor() {
		super();
		this.repository = Repository;
	}

	connectedCallback(): void {
		const requiredAttributes = [this.imageKeyAttribute];
		if (!hasRequiredAttributes(this, requiredAttributes)) {
			this.innerHTML = "";
			return;
		}

		const imageKey = this.getAttribute(this.imageKeyAttribute)!;
		this.imageKey = parseInt(imageKey);
		this.imageElementId = `model-image-${imageKey}`;
		this.labelElementId = `${this.imageElementId}-label`;

		this.innerHTML = this.render();
		this.bindEvents();
		this.repository
			.getImage<ClassificationResult>(this.imageKey)
			.then((image) => {
				this.modelImage = image;
				this.loadImage();
			});
	}

	private loadImage(): any {
		if (!this.modelImage) {
			return;
		}

		const imgElement = this.querySelector(
			`#${this.imageElementId}`
		) as HTMLImageElement;

		imgElement.onload = (ev) => this.getOrInferLabel(ev);
		imgElement.src = this.modelImage.imageData;
	}

	private getOrInferLabel(ev: Event) {
		if (!this.modelImage) {
			return;
		}

		const imgElement = ev.target as HTMLImageElement;
		const label = document.querySelector(
			`#${this.labelElementId}`
		) as HTMLLabelElement;

		if (this.modelImage.predictionResult) {
			label.innerHTML = this.modelImage.predictionResult.label;
		} else {
			label.innerHTML = "Running inference...";
			runInference(imgElement, "MobileNet").then((results) => {
				const inferredLabel = results[0].label;
				this.modelImage!.predictionResult = {
					label: inferredLabel,
				};
				this.repository.updateImageData(
					this.imageKey!,
					this.modelImage!
				);
				label.innerHTML = inferredLabel;
			});
		}
	}

	private bindEvents(): void {}

	private render(): string {
		return `
            <div>
                <img id="${this.imageElementId}" class="image, m-2, is-fullwidth" />
            </div>
            <div>
                <label id=${this.labelElementId}></label>
            </div>
        `;
	}
}

export class ImageUploader extends HTMLElement {
	private repository: ImageRepository;
	private imageIds: number[] = [];
	private objectStoreNameAttribute = "object-store-name";

	constructor() {
		super();
		this.repository = Repository;
	}

	connectedCallback(): void {
		if (!hasRequiredAttributes(this, [this.objectStoreNameAttribute])) {
			this.innerHTML = "";
			return;
		}

		this.innerHTML = this.render();
		this.bindEvents();
		this.repository
			.initializeDB(this.getAttribute(this.objectStoreNameAttribute)!)
			.then(() => this.loadImages());
	}

	// Load images from IndexedDB
	private loadImages(): void {
		this.repository.getAllIds().then((ids) => {
			this.imageIds = ids;
			this.updateImageDisplay();
		});
	}

	// Bind events for file upload and file changes
	private bindEvents(): void {
		const imageInput = this.querySelector(
			"#imageInput"
		) as HTMLInputElement;
		imageInput.addEventListener("change", (event: Event) => {
			const target = event.target as HTMLInputElement;
			if (target.files) {
				this.handleFiles(target.files);
				imageInput.value = "";
			}
		});
	}

	// Handle selected files
	private handleFiles(files: FileList): void {
		Array.from(files).forEach((file: File) => {
			const reader = new FileReader();
			reader.onload = (event: ProgressEvent<FileReader>) => {
				if (event.target && event.target.result) {
					const imageData = event.target.result as string;
					this.repository
						.addImage({
							imageData,
							metadata: {},
							stage: Stage.Inference,
						})
						.then(() => this.loadImages());
				}
			};
			reader.readAsDataURL(file);
		});
	}

	// Update the image display
	private async updateImageDisplay(): Promise<void> {
		const imageContainer = this.querySelector(
			"#imageContainer"
		) as HTMLElement;
		imageContainer.innerHTML = "";
		Promise.all(
			this.imageIds.map(async (id) => {
				const cell = document.createElement("div");
				cell.classList.add("cell");

				cell.innerHTML = `
                    <image-classification image-key="${id}" />
                `;

				imageContainer.appendChild(cell);
			})
		);
	}

	// Render the HTML structure
	private render(): string {
		return `
            <div>
                <label for="imageInput" class="button is-primary">Select Images</label>
                <input id="imageInput" accept="image/*" type="file" class="input" style="visibility:hidden;" multiple>
                <div class="block">
                    <div class="fixed-grid has-3-cols">
                        <div class="grid" id="imageContainer" />
                    <div>
                </div>
            </div>
      `;
	}
}

function hasRequiredAttributes(
	element: HTMLElement,
	attributes: string[]
): boolean {
	let result = true;

	for (const attribute of attributes) {
		if (!element.hasAttribute(attribute)) {
			console.error(
				`missing attribute in ${element.localName}: ${attribute}`
			);
			result = false;
		}
	}

	return result;
}
