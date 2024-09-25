import ort, { InferenceSession } from "onnxruntime-web";

import { createInferenceSession, runInference } from "modules";
import {
	ImageRepository,
	KeyValuePairs,
	ModelImage,
	ModelImageUpdateInput,
	Repository,
	Stage,
} from "modules/ImageRepository";

type InferenceInput<T extends KeyValuePairs> = {
	id: number;
	imgElement: HTMLImageElement;
	modelImage: ModelImage<T>;
	container: HTMLDivElement;
};

export abstract class VisionModelTrainerBase<
	TPredictionResult extends KeyValuePairs
> extends HTMLElement {
	private objectStoreNameAttribute = "object-store-name";
	private modelUrlAttribute = "model-url";
	private inputImageSizeAttribute = "input-image-size";
	private normRangeAtribute = "norm-range";

	private modelUrl: string = "";
	private inputImageSize: number = 224;
	private normRange: [number, number] = [-1, 1];

	private repository: ImageRepository;
	private modelImageIds: number[] = [];
	private columnSize: number = 12;

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

	protected abstract updatePredictionResult(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		predicitonResult: any
	): void;

	protected abstract updateInferenceLoading(
		container: Pick<HTMLDivElement, "querySelector">,
		imageId: number,
		message: string
	): void;

	protected abstract modelOutputsToPredictionResult(
		outputs: ort.InferenceSession.OnnxValueMapType,
		session: InferenceSession
	): TPredictionResult;

	protected async updateImageData(
		id: number,
		update: ModelImageUpdateInput<TPredictionResult>
	): Promise<void> {
		await this.repository.updateImageData(id, update);
	}

	protected async deleteImage(id: number) {
		await this.repository.deleteImage(id);
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

		const images = await this.repository.getAllImages();
		const inProgress = images.filter(
			(img) => img.stage === Stage.ReadyForTraining
		).length;
		const total = this.modelImageIds.length;
		progressElement.value = inProgress;
		progressElement.max = total;

		if (total === 0) {
			progressTextElement.innerText = `No images available`;
			startTrainBtn.disabled = true;
		} else if (inProgress === total) {
			progressTextElement.innerText = `All images reviewed!`;
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
				this.objectStoreNameAttribute,
				this.modelUrlAttribute,
				this.inputImageSizeAttribute,
				this.normRangeAtribute,
			])
		) {
			this.innerHTML = "";
			return;
		}

		this.modelUrl = this.getAttribute(this.modelUrlAttribute)!;
		this.inputImageSize = parseInt(
			this.getAttribute(this.inputImageSizeAttribute)!
		);
		const normRangeStr = this.getAttribute(this.normRangeAtribute)!;
		this.normRange = normRangeStr.split(",").map((n) => parseInt(n)) as [
			number,
			number
		];

		this.innerHTML = this.render();
		this.bindEvents();
		this.repository
			.initializeDB(this.getAttribute(this.objectStoreNameAttribute)!)
			.then(() => this.loadImages());
	}

	private loadImages(): void {
		this.repository.getAllIds().then(async (ids) => {
			this.modelImageIds = ids;
			this.updateImageDisplay();
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
		await this.runInference(renderedImages);
	}

	private async runInference(
		inferenceInput: InferenceInput<TPredictionResult>[]
	) {
		const inferenceSession = await createInferenceSession(this.modelUrl);
		for (const input of inferenceInput) {
			const { imgElement, id, modelImage, container } = input;
			if (!modelImage.predictionResult) {
				this.updateInferenceLoading(
					container,
					id,
					"Running Inference..."
				);
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
		await inferenceSession.release();
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
