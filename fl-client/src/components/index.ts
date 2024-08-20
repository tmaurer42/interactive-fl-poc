import { ImageInferenceElement } from "./ImageInference";
import { ImageUploader, ImageClassificationComponent } from "./ImageUploader";

export const registerComponents = () => {
	window.customElements.define("image-inference", ImageInferenceElement);
	window.customElements.define(
		"image-classification",
		ImageClassificationComponent
	);
	window.customElements.define("image-uploader", ImageUploader);
};
