import { hasRequiredAttributes } from "modules";
import {
	ImageRepository,
	KeyValuePairs,
	ModelImage,
	Repository,
	Stage,
} from "modules/ImageRepository";

export abstract class VisionDatasetDisplayBase<
	TPredictionResult extends KeyValuePairs
> extends HTMLElement {
	private taskIdAttribute = "task-id";

	private repository: ImageRepository;
	private columnSize: number = 12;
	private imageIds: number[] = [];

	constructor() {
		super();
		this.repository = Repository;
	}

	protected abstract headerText: string;

	protected abstract renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<TPredictionResult>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	): void;

	connectedCallback(): void {
		if (!hasRequiredAttributes(this, [this.taskIdAttribute])) {
			this.innerHTML = "";
			return;
		}

		this.innerHTML = this.render();
		this.bindEvents();
		this.repository
			.initializeDB(this.getAttribute(this.taskIdAttribute)!)
			.then(() => this.loadImages());
	}

	private bindEvents(): void {}

	private loadImages(): void {
		this.repository.getAllIds(Stage.Trained).then((ids) => {
			this.imageIds = ids;
			this.updateImageDisplay();
		});
	}

	private async updateImageDisplay(): Promise<void> {
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
				<div class="block is-flex is-flex-wrap-nowrap is-justify-content-space-between">
					<div>
						<div class="mt-2">${this.headerText}</div>
					</div>
				</div>
                <div class="block" style="height:calc(100vh - 270px);overflow-y:scroll">
                    <div class="grid is-col-min-${this.columnSize}" id="imageContainer" />
                </div>
            </div>
      `;
	}
}
