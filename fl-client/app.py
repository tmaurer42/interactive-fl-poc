import configparser
from datetime import datetime, timedelta
import io
from mimetypes import guess_type

import requests
from flask import Flask, make_response, render_template, send_file, send_from_directory


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

cfg_parser = configparser.ConfigParser()
cfg_parser.read('config.ini')
config = cfg_parser['DEFAULT']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test-model')
def test_model():
    model_id = "mobilenet_pretrained"
    resp = requests.get(f"{config['ServerUrl']}/api/global-model/{model_id}")
    model = resp.json()
    download_url = model['uri']
    input_size = model['input_size']
    norm_range = model['norm_range']

    return render_template('test_model.html', 
        model_url=download_url,
        input_size=input_size,
        scale_range=norm_range)


@app.route('/train-model')
def train_model():
    model_id = "mobilenet_pretrained_demo"
    resp = requests.get(f"{config['ServerUrl']}/api/global-model/{model_id}")
    model = resp.json()

    return render_template('train_model.html', model=model)


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/download/<path:filepath>', methods=['GET'])
def download(filepath: str):
    file_url = f"{config['ServerUrl']}/download/{filepath}"
    resp = requests.get(file_url)
    file_bytes: bytes = resp.content
    file_name = filepath.split('/')[-1]
    mime_type = guess_type(file_name)[0] or 'application/octet-stream'

    response = make_response(
        send_file(
            io.BytesIO(file_bytes),
            as_attachment=True,
            download_name=file_name,
            mimetype=mime_type
        )
    )

    last_modified = datetime.now()
    etag = f'{hash(file_bytes)}-{last_modified.timestamp()}'
    expires = last_modified + timedelta(seconds=3600)

    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['Last-Modified'] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response.headers['ETag'] = etag
    response.headers['Expires'] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")

    return response


if __name__ == '__main__':
    port = 4002
    app.run(debug=True, host='0.0.0.0', port=port)
