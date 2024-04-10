const path = require('path');

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
        path: path.resolve(__dirname, 'static/dist/js'),
        library: {
            type: 'umd'
        },
        sourceMapFilename: '[file].map',
    },
    devtool: 'source-map'
}