import { ImageInferenceElement } from "./ImageInference";
import { ImageUploader } from './ImageUploader'

export const registerComponents = () => {
	window.customElements.define("image-inference", ImageInferenceElement);
	window.customElements.define("image-uploader", ImageUploader);
};
