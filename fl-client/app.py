import configparser

import requests
from flask import Flask, render_template, send_from_directory


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

cfg_parser = configparser.ConfigParser()
cfg_parser.read('config.ini')
config = cfg_parser['DEFAULT']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/inference')
def inference():
    model_id = "mobilenet_pretrained"
    resp = requests.get(f"{config['ServerUrl']}/api/global-model/{model_id}")
    model = resp.json()
    model_download_url = model['uri']

    return render_template('inference.html')


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


port = 4002

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
