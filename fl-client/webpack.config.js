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
            {
                test: /\.js$/,
                enforce: "pre",
                use: ["source-map-loader"],
            },
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
        modules: [path.resolve(__dirname, 'src'), 'node_modules']
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
                { from: './node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
                { from: './node_modules/onnxruntime-web/dist/ort-training-wasm-simd-threaded.mjs', to: '[name][ext]' },
            ]
        })
    ],
    devtool: 'source-map'
}