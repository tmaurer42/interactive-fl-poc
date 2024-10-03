import {
	createInferenceSession,
	hasRequiredAttributes,
	ImageRepository,
	KeyValuePairs,
	ModelImage,
	ModelImageUpdateInput,
	Repository,
	Stage,
} from "modules";
import ort from "onnxruntime-web";

export type InferenceInput<T extends KeyValuePairs> = {
	id: number;
	imgElement: HTMLImageElement;
	modelImage: ModelImage<T>;
	container: HTMLDivElement;
};

export abstract class VisionModelTrainerBase<
	TPredictionResult extends KeyValuePairs
> extends HTMLElement {
	private taskIdAttribute = "task-id";
	private modelUrlAttribute = "model-url";

	private modelUrl: string = "";

	private repository: ImageRepository;
	private modelImageIds: number[] = [];
	private columnSize: number = 10;

	constructor() {
		super();
		this.repository = Repository;
	}

	protected abstract uploadButtonHintText: string;

	protected abstract renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<TPredictionResult>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	): void;

	protected abstract runInference(
		session: ort.InferenceSession,
		inferenceInput: InferenceInput<TPredictionResult>[]
	): Promise<void>;

	protected async updateImageData(
		id: number,
		update: ModelImageUpdateInput<TPredictionResult>
	): Promise<void> {
		await this.repository.updateImageData(id, update);
	}

	protected async deleteImage(id: number) {
		await this.repository.deleteImage(id);
		this.modelImageIds = this.modelImageIds.filter((imgId) => imgId !== id);
	}

	protected async updateProgressDisplay() {
		const progressElement = document.querySelector(
			"#reviewProgress"
		) as HTMLProgressElement;
		const progressTextElement = document.querySelector(
			"#reviewProgressText"
		) as HTMLDivElement;
		const startTrainBtn = document.querySelector(
			"#startTrainButton"
		) as HTMLButtonElement;

		const imagesReady = await this.repository.getAllImages(
			Stage.ReadyForTraining
		);
		const inProgress = imagesReady.length;
		const total = this.modelImageIds.length;
		progressElement.value = inProgress;
		progressElement.max = total;

		if (total === 0) {
			progressTextElement.innerText = `No images available`;
			startTrainBtn.disabled = true;
		} else if (inProgress === total) {
			progressTextElement.innerText = `All ${total} images reviewed!`;
			const startTrainBtn = document.querySelector(
				"#startTrainButton"
			) as HTMLButtonElement;
			startTrainBtn.disabled = false;
		} else {
			progressTextElement.innerText = `${inProgress}/${total} images reviewed`;
			startTrainBtn.disabled = true;
		}
	}

	connectedCallback(): void {
		if (
			!hasRequiredAttributes(this, [
				this.taskIdAttribute,
				this.modelUrlAttribute,
			])
		) {
			this.innerHTML = "";
			return;
		}

		this.modelUrl = this.getAttribute(this.modelUrlAttribute)!;

		this.innerHTML = this.render();
		this.bindEvents();
		this.repository
			.initializeDB(this.getAttribute(this.taskIdAttribute)!)
			.then(() => this.loadImages());
	}

	private loadImages(): void {
		this.repository.getAllIds(Stage.Inference).then((ids1) => {
			this.repository.getAllIds(Stage.ReadyForTraining).then((ids2) => {
				this.modelImageIds = [...ids1, ...ids2];
				this.updateImageDisplay();
			});
		});
	}

	private bindEvents(): void {
		const imageInput = this.querySelector(
			"#imageInput"
		) as HTMLInputElement;
		imageInput.onchange = (event: Event) => {
			const target = event.target as HTMLInputElement;
			if (target.files) {
				this.handleFiles(target.files);
				imageInput.value = "";
			}
		};

		const startTrainButton = this.querySelector(
			"#startTrainButton"
		) as HTMLButtonElement;
		startTrainButton.onclick = () => this.openTrainerModal();

		const trainerModal = document.querySelector("#trainer");
		trainerModal?.addEventListener("training-finished", () => {
			this.loadImages();
		});
	}

	private handleFiles(files: FileList): void {
		const readFileAsDataURL = (file: File): Promise<string> => {
			return new Promise<string>((resolve, reject) => {
				let reader = new FileReader();
				reader.onload = (e) => {
					if (e.target?.result) {
						resolve(e.target.result as string);
					} else {
						reject(new Error("Failed to read file as Data URL"));
					}
				};
				reader.onerror = () => reject(new Error("File reading error"));
				reader.readAsDataURL(file);
			});
		};

		const addImageToRepository = async (
			imageData: string
		): Promise<void> => {
			await this.repository.addImage({
				imageData,
				metadata: {},
				stage: Stage.Inference,
			});
		};

		const processFile = async (file: File): Promise<void> => {
			const imageData = await readFileAsDataURL(file);
			return await addImageToRepository(imageData);
		};

		Promise.all(Array.from(files).map(processFile)).then(() =>
			this.loadImages()
		);
	}

	private async updateImageDisplay(): Promise<void> {
		const imageContainer = this.querySelector(
			"#imageContainer"
		) as HTMLElement;
		imageContainer.innerHTML = "";

		const renderContentAndGetInferenceInput = async (
			id: number
		): Promise<InferenceInput<TPredictionResult>> => {
			const modelImage =
				await this.repository.getImage<TPredictionResult>(id);

			const cell = document.createElement("div");
			cell.classList.add("cell");

			return new Promise<InferenceInput<TPredictionResult>>((resolve) => {
				const onImageLoaded = (imgElement: HTMLImageElement) => {
					resolve({
						id,
						imgElement,
						modelImage: modelImage!,
						container: cell,
					});
				};
				imageContainer.appendChild(cell);
				this.renderImageCell(cell, id, modelImage!, onImageLoaded);
			});
		};

		const renderedImages = await Promise.all(
			this.modelImageIds.map(renderContentAndGetInferenceInput)
		);

		this.updateProgressDisplay();
		const inferenceSession = await createInferenceSession(this.modelUrl);
		await this.runInference(
			inferenceSession,
			renderedImages.filter((img) => !img.modelImage.predictionResult)
		);
		await inferenceSession.release();
	}

	private openTrainerModal() {
		const trainerModal = document.querySelector("#trainer");
		trainerModal?.setAttribute("is-active", "");
	}

	private render(): string {
		return `
            <div>
				<div class="block is-flex is-flex-wrap-nowrap is-justify-content-space-between">
					<div>
						<label for="imageInput" class="button is-primary">Select Images</label>
						<input id="imageInput" accept="image/*" type="file" class="input" style="display:none;" multiple>
						<div class="mt-2">${this.uploadButtonHintText}</div>
					</div>
					<div class="is-flex is-flex-wrap-nowrap">
						<div style="width:20em;">
							<div class="pb-2 pr-2 is-pulled-right">
								<span id="reviewProgressText"></span>
							</div>
							<div>
								<progress id="reviewProgress" class="progress is-primary is-medium is-dark"></progress>
							</div>
						</div>
						<div>
							<button id="startTrainButton" class="mt-2 ml-4 button is-info" disabled>Train</button>
						</div>
					</div>
				</div>
                <div class="block" style="height:calc(100vh - 320px);overflow-y:scroll">
                    <div class="grid is-col-min-${this.columnSize}" id="imageContainer" />
                </div>
            </div>
      `;
	}
}
