import {
	ClassificationModelTrainer,
	ClassificationTrainerModal,
	ClassificationDatasetDisplay,
	ImageClassificationCard,
	ClassificationModelTester,
} from "./imageClassification";

export const registerComponents = () => {
	window.customElements.define(
		"classification-trainer-modal",
		ClassificationTrainerModal
	);
	window.customElements.define("image-card", ImageClassificationCard);
	window.customElements.define(
		"classification-model-trainer",
		ClassificationModelTrainer
	);
	window.customElements.define(
		"classification-dataset-display",
		ClassificationDatasetDisplay
	);
	window.customElements.define(
		"classification-model-tester",
		ClassificationModelTester
	);
	window;
};
