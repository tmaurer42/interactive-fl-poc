from flask import request, jsonify
from functools import wraps


def check_registration(clients: set):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.data:
                client_id = request.get_json().get('clientId', '')
            else:
                client_id = request.headers.get('Client-Id', '')
            if client_id not in clients:
                return jsonify({'message': 'Client not registered'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
