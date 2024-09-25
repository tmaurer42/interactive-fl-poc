import os
import onnx
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import torch
import numpy as np

def main():
    eval_model = onnx.load("__file_storage__/models/mobilenet_pretrained_demo/eval_model.onnx")
    for node in eval_model.graph.node:
        if node.op_type == "BatchNormalization":
            del node.output[1:]
    onnx.save(eval_model, "__file_storage__/models/mobilenet_pretrained_demo/eval_model.onnx")

    state = CheckpointState.load_checkpoint("__file_storage__/models/mobilenet_pretrained_demo/checkpoint")

    # Create module.
    model = Module("__file_storage__/models/mobilenet_pretrained_demo/training_model.onnx", state, "__file_storage__/models/mobilenet_pretrained_demo/eval_model.onnx")
    # Create optimizer.
    optimizer = Optimizer("__file_storage__/models/mobilenet_pretrained_demo/optimizer_model.onnx", model)
    optimizer.set_learning_rate(0.1)
    state.save_checkpoint(model._state, "__file_storage__/models/mobilenet_pretrained_demo/checkpoint", include_optimizer_state=True)

    state = CheckpointState.load_checkpoint("__file_storage__/models/mobilenet_pretrained_demo/checkpoint")
    # Create module.
    model = Module("__file_storage__/models/mobilenet_pretrained_demo/training_model.onnx", state, "__file_storage__/models/mobilenet_pretrained_demo/eval_model.onnx")
    # Create optimizer.
    optimizer = Optimizer("__file_storage__/models/mobilenet_pretrained_demo/optimizer_model.onnx", model)
    lr = optimizer.get_learning_rate()

    model.train()
    input = torch.randn(2, 3, 224, 224).numpy()
    forward_inputs = [input,np.array([0,0])]
    foo = model(*forward_inputs)
    optimizer.step()
    model.lazy_reset_grad()

    model.eval()
    bar = model(*forward_inputs)
    pass

if __name__ == "__main__":
    main()