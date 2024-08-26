import { ImageInferenceElement } from "./ImageInference";
import { ClassificationImageUploader } from "./ClassificationImageUploader";

export const registerComponents = () => {
	window.customElements.define("image-inference", ImageInferenceElement);
	window.customElements.define(
		"classification-image-uploader",
		ClassificationImageUploader
	);
};
