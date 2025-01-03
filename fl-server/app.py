import io
from mimetypes import guess_type
import os
from typing import get_args
from flask import Flask, request, jsonify, send_file
from ml.ml_models import get_ml_model, MLModel
from ml.onnx_utils import model_to_onnx
from fl.fl_task import ClassificationFLTask, FLTaskBase, Aggregator
from repository.repository import InMemoryRepository
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
repository = InMemoryRepository[FLTaskBase]()

tasks: dict[str, FLTaskBase] = {}

def initialize_model(task: FLTaskBase, **model_args):
    model_dir = os.path.join(
        'models',
        task.id
    )
    model = get_ml_model(task.model, **model_args)
    trainable_param_names = [name for name,
                         p in model.named_parameters() if p.requires_grad]

    task.trainable_parameter_names = trainable_param_names
    task.model_file = os.path.join(model_dir, "model.onnx")
    task.training_file = os.path.join(model_dir, "training_model.onnx")
    task.optimizer_file = os.path.join(model_dir, "optimizer_model.onnx")
    task.eval_file = os.path.join(model_dir, "eval_model.onnx")
    task.checkpoint_file = os.path.join(model_dir, "checkpoint")

    model_to_onnx(
        model=model,
        model_directory=os.path.join('__file_storage__', model_dir)
    )


demo_model_id = 'demo_task'

task = ClassificationFLTask(
    id=demo_model_id,
    title='Task for demonstration',
    model='MobileNetV2',
    aggregator="fedasync",
    aggregator_params={'mixing_param': 0.5},
    classes=["Cat", "Dog"],
    local_epochs=10,
    batch_size=16,
    input_size=224,
    norm_range=[-1, 1]
)
initialize_model(task, num_classes=2)
repository.create(demo_model_id, task)


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
    task = repository.get(task_id)
    if task is None:
        return {'message': 'Task not found'}, 404

    return jsonify(vars(task))


@api.route('/api/metadata', methods=['GET'])
def get_metadata():
    models = list(get_args(MLModel))
    aggregators = list(get_args(Aggregator))
    return {
        'models': models,
        'aggregators': aggregators
    }, 200


@api.route('/api/tasks/<task_id>', methods=['PATCH'])
def update_task(task_id):
    task_update = request.json['task']
    print(task_update)
    if task_update is None:
        return {'message': 'No "task" provided in request body'}, 400

    task: FLTaskBase = repository.get(task_id)
    if task is None:
        return {'message': 'Task not found'}, 404

    for key,value in task_update.items():
        if key in vars(task):
            setattr(task, key, value)
        elif key in task.aggregator_params:
            task.aggregator_params[key] = value

    task.model_version = 0

    repository.update(task_id, task)
    initialize_model(task, num_classes=len(task.classes))

    return {'message': f'Task with id {task_id} updated successfully'}, 200


# Endpoint for clients to send their local model
@api.route('/api/model', methods=['POST'])
def update_model():
    task_id = request.json['task_id']
    update = request.json['update']
    model_version = request.json['model_version']

    task: FLTaskBase = repository.get(task_id)
    if task is None:
        return {'message': 'Task with id {taskId} not found'}, 404

    task.handleUpdate(
        update,
        model_version,
        storage,
    )

    repository.update(task_id, task)

    return {'message': f'Model for task update received'}, 200


port = 5002

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=port)
