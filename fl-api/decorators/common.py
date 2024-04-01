from flask import request, jsonify
from functools import wraps


def check_registration(clients):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.get_json().get('client_id')
            if client_id not in clients:
                return jsonify({'message': 'Client not registered'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
