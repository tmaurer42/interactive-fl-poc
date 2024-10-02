import configparser
from datetime import datetime, timedelta
import io
from mimetypes import guess_type

import requests
from flask import Flask, jsonify, make_response, render_template, request, send_file, send_from_directory


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

cfg_parser = configparser.ConfigParser()
cfg_parser.read('config.ini')
config = cfg_parser['DEFAULT']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train-model')
def train_model_get():
    task_id = "mobilenet_pretrained_demo"
    resp = requests.get(f"{config['ServerUrl']}/api/tasks/{task_id}")
    task = resp.json()

    return render_template('train_model.html', task=task)


@app.route('/train-model', methods=['POST'])
def train_model_post():
    data = request.get_json()
    
    task_id = data.get('task_id')
    update: dict[str, float] = data.get('update') 
    model_version: int = data.get('model_version')

    if task_id is None or update is None or model_version is None:
        return jsonify({'message': 'task_id, update and model_version are required'}), 400

    requests.post(f"{config['ServerUrl']}/api/model", json={
        'task_id': task_id,
        'update': update,
        'model_version': model_version
    })

    return jsonify({
        'message': f'Update received for task {task_id}'
    }), 200


@app.route('/my_dataset')
def my_dataset_get():
    task_id = "mobilenet_pretrained_demo"
    resp = requests.get(f"{config['ServerUrl']}/api/tasks/{task_id}")
    task = resp.json()

    return render_template('my_dataset.html', task=task)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/download/<path:filepath>', methods=['GET'])
def download(filepath: str):
    file_url = f"{config['ServerUrl']}/download/{filepath}"
    resp = requests.get(file_url)
    if resp.status_code != 200:
        return {'message': 'Download failed'}, resp.status_code
        
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

    """
    last_modified = datetime.now()
    etag = f'{hash(file_bytes)}-{last_modified.timestamp()}'
    expires = last_modified + timedelta(seconds=3600)

    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['Last-Modified'] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response.headers['ETag'] = etag
    response.headers['Expires'] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    """

    return response


if __name__ == '__main__':
    port = 4002
    app.run(debug=True, host='0.0.0.0', port=port)
