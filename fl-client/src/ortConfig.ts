import {env} from 'onnxruntime-web'

export const configureOrt = () => {
    env.wasm.wasmPaths = '/static/dist/';
}