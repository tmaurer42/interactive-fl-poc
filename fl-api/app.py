from flask import Flask, request, jsonify
from decorators.common import check_registration


api = Flask(__name__)

# Placeholder for model data
global_models = {}
clients = []
local_models = {}


@api.route('/')
def index():
    return jsonify({'secret_key': 'Hello World!!'})


# Client registration
@api.route('/api/register', methods=['POST'])
def register():
    client_id = request.json['client_id']
    clients.append(client_id)

    return {'message': f'Client {client_id} registered'}


# Get global model with id model_id
@api.route('/api/global-model', methods=['GET'])
@check_registration(clients)
def get_model():
    model_id = request.args.get('model_id')
    global_model = global_models.get(model_id, None)
    if global_model is None:
        return {'message': 'Model not found'}, 404

    return global_model


# Endpoint for clients to send their local model
@api.route('/api/local-model', methods=['POST'])
@check_registration(clients)
def update_model():
    data = request.json
    client_id = data['client_id']
    local_models[client_id] = data['model']

    return {'message': f'Model from client {client_id} received'}


# Endpoint for server to aggregate local models
@api.route('/api/aggregate-model', methods=['POST'])
def aggregate_model():
    model_id = request.json['model_id']
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
