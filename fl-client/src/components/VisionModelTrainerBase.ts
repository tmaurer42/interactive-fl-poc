import ort, { InferenceSession } from "onnxruntime-web";

import { createInferenceSession, runInference } from "modules";
import {
	ImageRepository,
	KeyValuePairs,
	ModelImage,
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
	private numColumns: number = 5;

	constructor() {
		super();
		this.repository = Repository;
	}

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
		message: string
	): void;

	protected abstract modelOutputsToPredictionResult(
		outputs: ort.InferenceSession.OnnxValueMapType,
		session: InferenceSession
	): TPredictionResult;

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
		imageInput.addEventListener("change", (event: Event) => {
			const target = event.target as HTMLInputElement;
			if (target.files) {
				this.handleFiles(target.files);
				imageInput.value = "";
			}
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
				this.renderImageCell(cell, id, modelImage!, onImageLoaded);
				imageContainer.appendChild(cell);
			});
		};

		const inferenceInput = await Promise.all(
			this.modelImageIds.map(renderContentAndGetInferenceInput)
		);
		await this.runInference(inferenceInput);
	}

	private async runInference(
		inferenceInput: InferenceInput<TPredictionResult>[]
	) {
		const inferenceSession = await createInferenceSession(this.modelUrl);
		for (const input of inferenceInput) {
			const { imgElement, id, modelImage, container } = input;
			if (!modelImage.predictionResult) {
				this.updateInferenceLoading(container, "Running Inference...");
				const outputs = await runInference(
					inferenceSession,
					imgElement,
					this.inputImageSize,
					this.normRange
				);
				const predicitonResult = this.modelOutputsToPredictionResult(
					outputs,
					inferenceSession
				);
				modelImage.predictionResult = predicitonResult;
				await this.repository.updateImageData(id, modelImage);
				this.updatePredictionResult(container, id, predicitonResult);
			}
		}
	}

	private render(): string {
		return `
            <div>
                <label for="imageInput" class="button is-primary">Select Images</label>
                <input id="imageInput" accept="image/*" type="file" class="input" style="visibility:hidden;" multiple>
                <div class="block">
                    <div class="fixed-grid has-${this.numColumns}-cols">
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
