import io
from mimetypes import guess_type
from flask import Flask, request, jsonify, send_file
from model.file import File
from model.fl_model_base import FLModel
from model.fl_model_fedbuff import FedBuffFLModel
from storage.file_system_storage import FileSystemStorage
from storage.file_storage_interface import IFileStorage


api = Flask(__name__)

##############
## Services ##
##############
storage: IFileStorage = FileSystemStorage()

##########
## Data ##
##########
global_models: dict[str, FLModel] = {
    'mobilenet_pretrained': FedBuffFLModel(
        title='MobileNet (pretrained)',
        file=File('models/mobilenet_pretrained.onnx'),
    ),
    'squeezenet_pretrained': FedBuffFLModel(
        title='SqueezeNet (pretrained)',
        file=File('models/squeezenet_pretrained.onnx'),
    )
}
clients = set()
local_models = {}


############
## Routes ##
############
@api.route('/')
def index():
    return jsonify({'secret_key': 'Hello World!!'})


@api.route('/download/<path:filepath>', methods=['GET'])
def download(filepath: str):
    file_bytes = storage.read(filepath)
    file_name = filepath.split('/')[-1]
    mime_type = guess_type(file_name)[0] or 'application/octet-stream'

    return send_file(
        io.BytesIO(file_bytes),
        as_attachment=True,
        download_name=file_name,
        mimetype=mime_type)


# Client registration
@api.route('/api/register', methods=['POST'])
def register():
    client_id = request.json['clientId']
    clients.add(client_id)

    return {'message': f'Client {client_id} registered'}


# Get global model
@api.route('/api/global-model/<model_id>', methods=['GET'])
def get_model(model_id):
    global_model: FLModel = global_models.get(model_id, None)
    if global_model is None:
        return {'message': 'Model not found'}, 404

    return {
        'title': global_model.title,
        'uri': f'/download/{global_model.file.path}',
    }


# Endpoint for clients to send their local model
@api.route('/api/local-model', methods=['POST'])
def update_model():
    data = request.json
    client_id = data['clientId']
    local_models[client_id] = data['model']

    return {'message': f'Model from client {client_id} received'}


# Endpoint for server to aggregate local models
@api.route('/api/aggregate-model', methods=['POST'])
def aggregate_model():
    model_id = request.json['modelId']
    global_model = global_models.get(model_id, None)
    if global_model is None:
        return {'message': 'Model not found'}, 404

    for client_id, model in local_models.items():
        # Aggregation logic
        pass

    return {'message': 'Global model updated'}


port = 5002

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=port)
