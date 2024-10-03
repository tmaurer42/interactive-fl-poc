import { hasRequiredAttributes, ImageRepository, Repository } from "modules";

type VisionTrainerModalBaseProps = {
	taskId: string;
	modelVersion: number;
	isActive: boolean;
	nEpochs: number;
	batchSize: number;
	trainingModelUrl: string;
	optimizerModelUrl: string;
	evalModelUrl: string;
	checkpointUrl: string;
};

export abstract class VisionTrainerModalBase<TProps> extends HTMLElement {
	static observedAttributes = ["is-active"];

	private taskIdAttribute = "task-id";
	private modelVersionAttribute = "model-version";
	private isActiveAttribute = "is-active";
	private nEpochsAttribute = "n-epochs";
	private batchSizeAttribute = "batch-size";
	private trainingModelUrlAttribute = "training-model-url";
	private optimizerModelUrlAttribute = "optimizer-model-url";
	private evalModelUrlAttribute = "eval-model-url";
	private checkpointUrlAttribute = "checkpoint-url";

	protected properties = {} as VisionTrainerModalBaseProps & TProps;
	protected repository: ImageRepository;

	constructor() {
		super();
		this.repository = Repository;
	}

	protected abstract train(
		onTrainStart: () => void,
		updateTrainingProgress: (epoch: number, loss: number) => void,
		sendUpdate: (params: Float32Array) => Promise<void>,
		onTrainEnd: () => void,
		updateProgressMessage: (message: string) => void
	): Promise<void>;

	protected updateProgressMessage(message: string) {
		const progressMessageSpan = this.querySelector(
			"#progress-message"
		) as HTMLSpanElement;
		progressMessageSpan.innerText = message;
	}

	connectedCallback(): void {
		if (
			!hasRequiredAttributes(this, [
				this.taskIdAttribute,
				this.modelVersionAttribute,
				this.nEpochsAttribute,
				this.batchSizeAttribute,
				this.trainingModelUrlAttribute,
				this.optimizerModelUrlAttribute,
				this.evalModelUrlAttribute,
				this.checkpointUrlAttribute,
			])
		) {
			this.innerHTML = "";
			return;
		}

		this.properties = {
			...this.properties,
			taskId: this.getAttribute(this.taskIdAttribute)!,
			modelVersion: parseInt(
				this.getAttribute(this.modelVersionAttribute)!
			),
			isActive: this.hasAttribute(this.isActiveAttribute),
			nEpochs: parseInt(this.getAttribute(this.nEpochsAttribute)!),
			batchSize: parseInt(this.getAttribute(this.batchSizeAttribute)!),
			trainingModelUrl: this.getAttribute(
				this.trainingModelUrlAttribute
			)!,
			optimizerModelUrl: this.getAttribute(
				this.optimizerModelUrlAttribute
			)!,
			evalModelUrl: this.getAttribute(this.evalModelUrlAttribute)!,
			checkpointUrl: this.getAttribute(this.checkpointUrlAttribute)!,
		};

		this.repository.initializeDB(this.properties.taskId).then(async () => {
			this.innerHTML = this.render();
			this.bindEvents();
		});
	}

	attributeChangedCallback(name: string, oldValue: string, newValue: string) {
		if (name === this.isActiveAttribute) {
			const modal = this.querySelector(
				"#trainer-modal"
			) as HTMLDivElement;
			if (this.hasAttribute(this.isActiveAttribute)) {
				this.repository.getAllIds().then((ids) => {
					const numImagesSpan = this.querySelector(
						"#num-images"
					) as HTMLSpanElement;
					numImagesSpan.innerText = ids.length.toString();
					modal.classList.add("is-active");
				});
			} else {
				modal.classList.remove("is-active");
			}
		}

		if (name === this.modelVersionAttribute) {
			this.properties.modelVersion = parseInt(newValue);
		}
	}

	private bindEvents() {
		const cancelButton = this.querySelector("#cancelBtn") as HTMLDivElement;
		const trainButton = this.querySelector(
			"#trainBtn"
		) as HTMLButtonElement;

		cancelButton.onclick = () => {
			this.removeAttribute("is-active");
			trainButton.disabled = false;
		};

		trainButton.onclick = () => {
			this.train(
				this.onTrainStart.bind(this),
				this.updateTrainingProgress.bind(this),
				this.sendUpdate.bind(this),
				this.onTrainEnd.bind(this),
				this.updateProgressMessage.bind(this)
			).then(() => {
				this.dispatchEvent(new CustomEvent("training-finished"));
			});
		};
	}

	private onTrainStart() {
		window.onbeforeunload = function (event) {
			event.preventDefault();
			return "";
		};

		const trainButton = this.querySelector(
			"#trainBtn"
		) as HTMLButtonElement;
		trainButton.disabled = true;

		const cancelButton = this.querySelector(
			"#cancelBtn"
		) as HTMLButtonElement;
		cancelButton.disabled = true;
	}

	private onTrainEnd() {
		window.onbeforeunload = null;

		const cancelButton = this.querySelector(
			"#cancelBtn"
		) as HTMLButtonElement;
		cancelButton.disabled = false;
	}

	private updateTrainingProgress(epochsTrained: number, loss: number) {
		const epochsTrainedElem = this.querySelector(
			"#epochs-trained"
		) as HTMLSpanElement;
		epochsTrainedElem.innerText = epochsTrained.toString();

		const lossValueElem = this.querySelector(
			"#loss-value"
		) as HTMLSpanElement;
		lossValueElem.innerText = loss.toFixed(3);

		const trainProgressElem = this.querySelector(
			"#train-progress"
		) as HTMLProgressElement;
		trainProgressElem.value = +(
			(epochsTrained / this.properties.nEpochs) *
			100
		).toFixed(2);
	}

	private async sendUpdate(newParams: Float32Array) {
		const { taskId, modelVersion } = this.properties;
		const update = Array.from(newParams);

		console.log(update);
		const data = {
			task_id: taskId,
			update: update,
			model_version: modelVersion,
		};

		await fetch(window.location.toString(), {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});
	}

	private render() {
		return `
            <div id="trainer-modal" class="modal">
                <div class="modal-background"></div>
                <div class="modal-card">
                    <header class="modal-card-head">
                        <p class="modal-card-title">Model Training</p>
                    </header>
                    <section class="modal-card-body">
                        <div class="block">
                            <div>Total number of images: <span id="num-images"></span></div>
                        </div>
                        <div class="block">
                            <div>Batch Size: ${this.properties.batchSize}</div>
                            <div>Learning Rate: 0.001</div>
                            <div>Epochs: ${this.properties.nEpochs}</div>
                        </div>
                        <div class="block">
                            <div class="is-flex is-justify-content-space-between mr-2 mb-2">
                                <div>
                                    <span id="progress-message"></span>
                                </div>
                                <div>
                                    Epoch <span id="epochs-trained">0</span>/${this.properties.nEpochs}
                                </div>
                            </div>
                            <div>
                                <progress id="train-progress" class="progress is-link" value="0" max="100" />
                            </div>
                        </div>
                        <div class="block is-flex is-justify-content-center">
                            <span class="pl-4 pr-2">
                                <strong>Loss: <span id="loss-value">0.00</span></strong>
                            </span>
                        </div>
                    </section>
                    <footer class="modal-card-foot">
                        <div class="buttons">
                            <button id="trainBtn" class="button is-success">Start Training Process</button>
                            <button id="cancelBtn" class="button">Close Dialog</button>
                        </div>
                    </footer>
                </div>
            </div>
        `;
	}
}
