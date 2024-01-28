import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CNN, RNN, CRNN

def get_model(model_name, num_classes):
    """
    Get the specified model by name.

    Args:
        model_name (str): Name of the model to retrieve.
        num_classes (int): Number of classes for the model.

    Returns:
        torch.nn.Module: The specified neural network model.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    models = {
        'cnn': CNN,
        'rnn': RNN,
        'crnn': CRNN
    }
    try:
        model = models[model_name](num_classes)
    except KeyError:
        raise ValueError(f'Invalid model name: {model_name}')
    return mode
