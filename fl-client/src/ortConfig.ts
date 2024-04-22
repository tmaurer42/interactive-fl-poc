import { env } from "onnxruntime-web";
import { env as webGPUEnv } from "onnxruntime-web/webgpu";

export const configureOrt = () => {
	env.wasm.wasmPaths = "/static/dist/";
	webGPUEnv.wasm.wasmPaths = "/static/dist/";
};
