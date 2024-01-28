import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import load_mfcc_features, save_mfcc_features, audio_preprocessing, audio_postprocessing, save_audio, load_audio, reduce_noise, extract_mfcc
from model import SpeechRecognitionCNN

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    epoch_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).tolist())
    return epoch_loss / len(val_loader), accuracy_score(y_true, y_pred)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, log_dir=None):
    """
    Train the model.
    """
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}')
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

    if writer is not None:
        writer.flush()
        writer.close()
    return model

def predict(model, test_loader, device):
    """
    Make predictions on the test set.
    """
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.argmax(dim=1).tolist())
    return y_pred

def save_predictions(y_pred, file_path):
    """
    Save predictions to a file.
    """
    df = pd.DataFrame(y_pred, columns=['label'])
    df.to_csv(file_path, index=False)

def load_predictions(file_path):
    """
    Load predictions from a file.
    """
    df = pd.read_csv(file_path)
    return df['label'].tolist()

def plot_confusion_matrix(y_true, y_pred, labels, file_path):
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(file_path)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_classification_report(y_true, y_pred, labels, file_path):
    """
    Plot a classification report.
    """
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_path)
    df = df.drop(columns=['support'])
    df = df.drop(index=['accuracy', 'macro avg', 'weighted avg'])
    df.plot.bar(rot=0)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
