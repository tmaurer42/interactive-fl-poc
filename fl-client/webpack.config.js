const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
    entry: path.resolve(__dirname, 'src/index.ts'),
    module: {
        rules: [
        {
            test: /\.ts?$/,
            use: 'ts-loader',
            exclude: /node_modules/,
        },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    output: {
        filename: 'index.js',
        path: path.resolve(__dirname, 'static/dist'),
        library: {
            type: 'umd'
        },
        sourceMapFilename: '[file].map',
    },
    plugins: [
        new CopyPlugin({
            patterns: [
                './node_modules/onnxruntime-web/dist/ort-wasm.wasm',       
                './node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
            ]
        })
    ],
    devtool: 'source-map'
}