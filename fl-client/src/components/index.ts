import { ImageInferenceElement } from "./ImageInference";

export const registerComponents = () => {
	window.customElements.define("image-inference", ImageInferenceElement);
};
