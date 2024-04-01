import pytest
from flask import Flask, jsonify
from decorators.common import check_registration

app = Flask(__name__)

registered_clients = ['client1']


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_check_registration_decorator(client):
    @app.route('/test', methods=['POST'])
    @check_registration(registered_clients)
    def test_route():
        return jsonify({'message': 'Success'}), 200

    # Test with a registered client
    response = client.post('/test', json={'client_id': 'client1'})
    assert response.status_code == 200
    assert response.get_json()['message'] == 'Success'

    # Test with a non-registered client
    response = client.post('/test', json={'client_id': 'client2'})
    assert response.status_code == 403
