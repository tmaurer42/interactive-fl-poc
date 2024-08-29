import * as ort from "onnxruntime-web";
import * as ortTrain from "onnxruntime-web/training";
import * as ortWebGPU from "onnxruntime-web/webgpu";

export const configureOrt = () => {
	ort.env.wasm.wasmPaths = "/static/dist/";
	ort.env.wasm.numThreads = 1;
	ortTrain.env.wasm.wasmPaths = "/static/dist/";
	ortTrain.env.wasm.numThreads = 1;
	ortWebGPU.env.wasm.wasmPaths = "/static/dist/";
	ortWebGPU.env.wasm.numThreads = 1;
};
