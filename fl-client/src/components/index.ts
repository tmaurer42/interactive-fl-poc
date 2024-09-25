import { ImageInferenceElement } from "./ImageInference";
import { ClassificationModelTrainer } from "./ClassificationModelTrainer";
import { ImageClassificationCard } from "./ImageCard";

export const registerComponents = () => {
	window.customElements.define("image-card", ImageClassificationCard);
	window.customElements.define("image-inference", ImageInferenceElement);
	window.customElements.define(
		"classification-model-trainer",
		ClassificationModelTrainer
	);
};
