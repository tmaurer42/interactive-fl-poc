import io
from mimetypes import guess_type
import os
from flask import Flask, request, jsonify, send_file
from ml.mobilenet import get_mobilenet
from ml.onnx_utils import model_to_onnx
from fl.fl_task import ClassificationFLTask, FLTaskBase
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
tasks: dict[str, FLTaskBase] = {}

demo_model_id = 'mobilenet_pretrained_demo'
demo_model_dir_internal = os.path.join(
    '__file_storage__',
    'models',
    demo_model_id
)
mobilenet = get_mobilenet(
    num_classes=2, transfer_learning=True, dropout=0.2)
trainable_param_names = [name for name,
                         p in mobilenet.named_parameters() if p.requires_grad]

if not os.path.exists(demo_model_dir_internal):
    model_to_onnx(
        model=mobilenet,
        model_directory=demo_model_dir_internal
    )

demo_model_dir = os.path.join('models', demo_model_id)
task = ClassificationFLTask(
    id='mobilenet_pretrained_demo',
    title='MobileNet (pretrained) for cats and dogs',
    aggregator="fedasync",
    aggregator_params={'mixing_param': 0.5},
    classes=["Cat", "Dog"],
    model_version=0,
    model_file=os.path.join(demo_model_dir, "model.onnx"),
    training_file=os.path.join(demo_model_dir, "training_model.onnx"),
    optimizer_file=os.path.join(demo_model_dir, "optimizer_model.onnx"),
    eval_file=os.path.join(demo_model_dir, "eval_model.onnx"),
    checkpoint_file=os.path.join(demo_model_dir, "checkpoint"),
    local_epochs=10,
    batch_size=16,
    input_size=224,
    norm_range=[-1, 1]
)
task.trainable_parameter_names = trainable_param_names
tasks[demo_model_id] = task


############
## Routes ##
############
@api.route('/')
def index():
    return jsonify({'secret_key': 'Hello World!!'})


# Download endpoint for anything stored in the storage
@api.route('/download/<path:filepath>', methods=['GET'])
def download(filepath: str):
    file_bytes = storage.read(filepath)
    file_name = filepath.split('/')[-1]
    mime_type = guess_type(file_name)[0] or 'application/octet-stream'

    return send_file(
        io.BytesIO(file_bytes),
        as_attachment=True,
        download_name=file_name,
        mimetype=mime_type
    )


# Get an FL task
@api.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    task = tasks.get(task_id, None)
    if task is None:
        return {'message': 'Task not found'}, 404

    return jsonify(vars(task))


# Endpoint for clients to send their local model
@api.route('/api/model', methods=['POST'])
def update_model():
    task_id = request.json['task_id']
    update = request.json['update']
    model_version = request.json['model_version']

    task = tasks.get(task_id, None)
    if task is None:
        return {'message': 'Task with id {taskId} not found'}, 404

    task.handleUpdate(
        update,
        model_version,
        storage,
    )

    return {'message': f'Model for task update received'}


port = 5002

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=port)
