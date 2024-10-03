import {
	hasRequiredAttributes,
	ImageRepository,
	KeyValuePairs,
	ModelImage,
	Repository,
	Stage,
} from "modules";

export abstract class VisionDatasetDisplayBase<
	TPredictionResult extends KeyValuePairs
> extends HTMLElement {
	private taskIdAttribute = "task-id";

	protected repository: ImageRepository;
	protected imageIds: number[] = [];

	protected columnSize: number = 8;
	protected stage?: Stage = undefined;

	constructor() {
		super();
		this.repository = Repository;
	}

	/**
	 * Render the cell for one image into the provided container.
	 * When the image has loaded, onImageLoaded needs to be called.
	 * @param container The container in which to render the image component.
	 * @param imageId Database id of the image.
	 * @param modelImage Data of the image.
	 * @param onImageLoaded Callback when image has loaded in the DOM.
	 */
	protected abstract renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<TPredictionResult>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	): void;

	/**
	 * Determines what to render in the header area
	 */
	protected abstract renderHeader(): string;

	connectedCallback(): void {
		if (!hasRequiredAttributes(this, [this.taskIdAttribute])) {
			this.innerHTML = "";
			return;
		}

		this.innerHTML = this.render();
		this.repository
			.initializeDB(this.getAttribute(this.taskIdAttribute)!)
			.then(() => this.loadImages());
	}

	/**
	 * Load all images associated with this component.
	 * After loading, calls updateImageDisplay().
	 */
	protected loadImages(): void {
		this.repository.getAllIds(this.stage).then((ids) => {
			this.imageIds = ids;
			this.updateImageDisplay();
		});
	}

	/**
	 * Render all images currently stored in the component.
	 */
	protected async updateImageDisplay(): Promise<void> {
		const imageContainer = this.querySelector(
			"#imageContainer"
		) as HTMLElement;
		imageContainer.innerHTML = "";

		const renderContent = async (id: number): Promise<void> => {
			const modelImage =
				await this.repository.getImage<TPredictionResult>(id);

			const cell = document.createElement("div");
			cell.classList.add("cell");

			return new Promise<void>((resolve) => {
				const onImageLoaded = (imgElement: HTMLImageElement) => {
					resolve();
				};
				imageContainer.appendChild(cell);
				this.renderImageCell(cell, id, modelImage!, onImageLoaded);
			});
		};

		Promise.all(this.imageIds.map(renderContent));
	}

	private render(): string {
		return `
            <div>
				${this.renderHeader()}
                <div class="block" style="height:calc(100vh - 270px);overflow-y:scroll">
                    <div class="grid is-col-min-${
						this.columnSize
					}" id="imageContainer" />
                </div>
            </div>
      `;
	}
}
