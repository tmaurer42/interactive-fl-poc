import * as inference from "modules/inference";
import { ImageRepository, ModelImage, Stage } from "modules/repository";

export class ImageUploader extends HTMLElement {
	private repository: ImageRepository | undefined;
	private images: ModelImage<any>[] = [];
	private objectStoreNameAttribute = "object-store-name";

	constructor() {
		super();
	}

	connectedCallback(): void {
		if (!this.hasAttribute(this.objectStoreNameAttribute)) {
			console.error(
				`missing attribute in ${ImageUploader.name}: ${this.objectStoreNameAttribute}`
			);
			this.innerHTML = "";
			return;
		}

		this.repository = new ImageRepository(
			this.getAttribute(this.objectStoreNameAttribute)!
		);
		this.innerHTML = this.render();
		this.repository.initializeDB().then(() => {
			this.loadImages();
		});
		this.bindEvents();
	}

	// Load images from IndexedDB
	private loadImages(): void {
		this.repository?.getAllImages().then((images) => {
			this.images = images;
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
						?.addImage({
							imageData,
							metadata: {},
							predictionResult: {},
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
			this.images.map(async (image) => {
				const cell = document.createElement("div");
				cell.classList.add("cell");
				const img = document.createElement("img");
				img.src = image.imageData;
				img.classList.add("image", "m-2", "is-fullwidth");
				const labelElement = document.createElement("label");
				labelElement.innerHTML = "Label";

				cell.appendChild(img);
				cell.appendChild(labelElement);
				imageContainer.appendChild(cell);

				// const inferenceResult = await runInference(img, "MobileNet");
				// const label = inferenceResult[0].label;
				// labelElement.innerHTML = label;
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
