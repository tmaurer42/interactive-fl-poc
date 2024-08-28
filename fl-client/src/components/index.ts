import { ImageInferenceElement } from "./ImageInference";
import { ClassificationModelTrainer } from "./ClassificationImageUploader";

export const registerComponents = () => {
	window.customElements.define("image-inference", ImageInferenceElement);
	window.customElements.define(
		"classification-model-trainer",
		ClassificationModelTrainer
	);
};
