import torch
import Levenshtein as lev
from torch.utils.data import DataLoader

def calculate_wer(predicted_texts, target_texts):
    """
    Calculate the Word Error Rate (WER).

    Args:
        predicted_texts (list of str): Predicted transcriptions.
        target_texts (list of str): Ground truth transcriptions.

    Returns:
        float: The average WER.
    """
    total_wer = 0.0
    for pred, target in zip(predicted_texts, target_texts):
        total_wer += lev.distance(pred.split(), target.split()) / len(target.split())

    return total_wer / len(target_texts)

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): Device on which to evaluate.

    Returns:
        float: The average WER on the dataset.
    """
    model.eval()
    predicted_texts = []
    target_texts = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)

            # Assume model's output is text. Modify according to your model's design
            outputs = model(inputs)

            predicted_texts.extend(outputs)
            target_texts.extend(labels)

    return calculate_wer(predicted_texts, target_texts)

# Example usage
# Assume model, test_loader, and device are predefined
# wer = evaluate_model(model, test_loader, device)
# print(f"Word Error Rate: {wer:.2f}")
