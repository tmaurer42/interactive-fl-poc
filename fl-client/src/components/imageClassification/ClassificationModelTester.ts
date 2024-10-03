import { VisionDatasetDisplayBase } from "components/base";
import {
	Stage,
	hasRequiredAttributes,
	ModelImage,
	createInferenceSession,
	batchify,
	preprocessImagesFromBase64,
	reshapeArray,
	softmax,
	classesTopK,
} from "modules";
import { ClassificationResult } from "./ClassificationResult";

export class ClassificationModelTester extends VisionDatasetDisplayBase<ClassificationResult> {
	private classesAttribute = "classes";
	private modelUrlAttribute = "model-url";
	private inputImageSizeAttribute = "input-image-size";
	private normRangeAtribute = "norm-range";

	private classes: string[] = [];
	private modelUrl = "";
	private selectedClass: string = "";
	private inputImageSize: number = 224;
	private normRange: [number, number] = [-1, 1];

	protected stage = Stage.Testing;

	connectedCallback(): void {
		if (
			!hasRequiredAttributes(this, [
				this.classesAttribute,
				this.modelUrlAttribute,
				this.inputImageSizeAttribute,
				this.normRangeAtribute,
			])
		) {
			this.innerHTML = "";
			return;
		}

		this.classes = this.getAttribute(this.classesAttribute)!.split(",");
		this.modelUrl = this.getAttribute(this.modelUrlAttribute)!;
		this.selectedClass = this.classes[0];
		this.inputImageSize = parseInt(
			this.getAttribute(this.inputImageSizeAttribute)!
		);
		this.normRange = this.getAttribute(this.normRangeAtribute)!
			.split(",")
			.map((n) => parseInt(n)) as [number, number];

		// Render the base component
		super.connectedCallback();
		this.bindEvents();
	}

	private bindEvents() {
		const imageInput = this.querySelector(
			"#image-input"
		) as HTMLInputElement;
		imageInput.onchange = (event: Event) => {
			const target = event.target as HTMLInputElement;
			if (target.files) {
				this.handleFiles(target.files);
				imageInput.value = "";
			}
		};

		const classDropdown = this.querySelector(
			"#class-select"
		) as HTMLSelectElement;
		classDropdown.onchange = () => {
			this.selectedClass = classDropdown.value;
		};

		const runTestButton = this.querySelector(
			"#run-test-btn"
		) as HTMLButtonElement;
		runTestButton.onclick = () => {
			this.runTest();
		};
	}

	private handleFiles(files: FileList) {
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
				predictionResult: {
					label: this.selectedClass,
				},
				stage: Stage.Testing,
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

	private onTestStart() {
		const imageInputElement = this.querySelector(
			"#image-input"
		) as HTMLInputElement;
		const classDropdown = this.querySelector(
			"#class-select"
		) as HTMLSelectElement;
		const runTestButton = this.querySelector(
			"#run-test-btn"
		) as HTMLButtonElement;

		imageInputElement.disabled = true;
		runTestButton.disabled = true;
		classDropdown.disabled = true;
	}

	private onTestEnd() {
		const imageInputElement = this.querySelector(
			"#image-input"
		) as HTMLInputElement;
		const classDropdown = this.querySelector(
			"#class-select"
		) as HTMLSelectElement;
		const runTestButton = this.querySelector(
			"#run-test-btn"
		) as HTMLButtonElement;

		imageInputElement.disabled = false;
		runTestButton.disabled = false;
		classDropdown.disabled = false;
	}

	private updateTestProgressMessage(message: string) {
		const progressMessageSpan = this.querySelector(
			"#test-progress-message"
		) as HTMLSpanElement;
		progressMessageSpan.innerText = message;
	}

	private async runTest() {
		this.onTestStart();
		this.updateTestProgressMessage("Running test session");

		const testBatchSize = 16;
		const testIds = await this.repository.getAllIds(Stage.Testing);
		const batches = batchify(testIds, testBatchSize);

		const inferenceSession = await createInferenceSession(this.modelUrl);

		const correctIds: number[] = [];
		const incorrectIds: number[] = [];
		let idIndex = 0;
		let batchNo = 1;

		for (const batch of batches) {
			this.updateTestProgressMessage(
				`Running test session: Batch ${batchNo}/${Math.ceil(
					testIds.length / testBatchSize
				)}`
			);

			const testData =
				await this.repository.getImagesByIds<ClassificationResult>(
					batch
				);
			const images = testData.map((d) => d!.imageData);

			const input = await preprocessImagesFromBase64(
				images,
				this.inputImageSize,
				this.normRange
			);

			const inputName = inferenceSession.inputNames[0];
			const result = await inferenceSession.run({
				[inputName]: input,
			});
			const output = result[inferenceSession.outputNames[0]];
			const outputShape = output.dims as number[];
			const imageScores = reshapeArray(outputShape, output.data);

			for (let i = 0; i < imageScores.length; i++) {
				const scores = imageScores[i];
				const outputSoftmax = softmax(
					Array.prototype.slice.call(scores)
				);
				const predictedLabel = classesTopK(
					outputSoftmax,
					Math.min(5, this.classes.length),
					this.classes
				)[0].label;
				const image = testData[i]!;
				const actualLabel = image.predictionResult?.label!;

				if (predictedLabel === actualLabel) {
					correctIds.push(testIds[idIndex]);
				} else {
					incorrectIds.push(testIds[idIndex]);
				}

				idIndex += 1;
			}

			batchNo += 1;
		}

		inferenceSession.release();

		for (const id of correctIds) {
			const labelElement = this.querySelector(
				`#image-label-${id}`
			) as HTMLSpanElement;
			labelElement.classList.add("has-text-success");
		}

		for (const id of incorrectIds) {
			const labelElement = this.querySelector(
				`#image-label-${id}`
			) as HTMLSpanElement;
			labelElement.classList.add("has-text-danger");
		}

		const acc = (correctIds.length / testIds.length) * 100;
		this.updateTestProgressMessage(
			`Accuracy: ${acc.toFixed(2)} % (Correct: ${
				correctIds.length
			}, Incorrect: ${incorrectIds.length})`
		);
		this.onTestEnd();
	}

	protected renderImageCell(
		container: HTMLDivElement,
		imageId: number,
		modelImage: ModelImage<ClassificationResult>,
		onImageLoaded: (imgElement: HTMLImageElement) => void
	): void {
		const label = modelImage.predictionResult?.label ?? "";
		const imgElement = document.createElement("img");
		imgElement.style.maxHeight = "100%";
		imgElement.id = `img-${imageId}`;
		imgElement.onload = () => onImageLoaded?.(imgElement);
		imgElement.src = modelImage.imageData;

		const imageHeight = 10;
		const imageCard = `
            <div class="card" id="image-card-${imageId}">
				<div>
					<button 
						id="btn-delete-img-${imageId}" 
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
							<span id="image-label-${imageId}">${label}</span>
						</strong>
					</label>
                </footer>
            </div>
        `;

		container.innerHTML = imageCard;

		const deleteImgButton = this.querySelector(
			`#btn-delete-img-${imageId}`
		) as HTMLButtonElement;

		deleteImgButton.onclick = () => {
			this.repository.deleteImage(imageId);
			container.remove();
		};
	}

	protected renderHeader(): string {
		return `
            <div id="tester-header" class="block is-flex is-flex-wrap-nowrap is-justify-content-space-between">
				<div class="is-flex is-flex-wrap-nowrap">
					<div class="mr-2 mb-2">
						<label id="image-input-label" for="image-input" class="button is-primary">Select Images for class</label>
						<input id="image-input" accept="image/*" type="file" class="input" style="display:none;" multiple>
					</div>
					<div class="select is-primary">
						<select id="class-select">
							${this.classes.map((cls) => `<option value="${cls}">${cls}</option>`)}
						</select>
					</div>
				</div>
				<div>
					<strong><span class="is-size-5" id="test-progress-message"></span></strong>
				</div>
				<div>
					<button id="run-test-btn" class="button is-info">Run Test</button>
				</div>
            </div>
        `;
	}
}
