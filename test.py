import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os

from model import get_model
from dataset_process import prepare_dataset, get_features, get_piano_roll


class MIDIDataset(Dataset):
    """
    PyTorch Dataset for MIDI data.
    """
    def __init__(self, features, labels, piano_rolls=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.piano_rolls = torch.FloatTensor(piano_rolls) if piano_rolls is not None else None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.piano_rolls is not None:
            return self.features[idx], self.piano_rolls[idx], self.labels[idx]
        else:
            return self.features[idx], self.labels[idx]


def test_model(model, dataloader, device, model_type):
    """
    Test the model and calculate accuracy.
    
    @input model: Trained model
    @type model: nn.Module
    @input dataloader: Test data loader
    @type dataloader: DataLoader
    @input device: Device to run on
    @type device: torch.device
    @input model_type: Type of model
    @type model_type: str
    
    @return: Test accuracy
    @rtype: float
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            if model_type == 'handcrafted':
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
            else:
                features, piano_rolls, labels = batch_data
                features = features.to(device)
                piano_rolls = piano_rolls.to(device)
                labels = labels.to(device)
                
                if model_type == 'piano_roll':
                    outputs = model(piano_rolls)
                else:  # combined models
                    outputs = model(features, piano_rolls)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def load_and_test(model_path,
                 genre_path="msd_tagtraum_cd1.cls",
                 midi_path="lmd_matched",
                 batch_size=32):
    """
    Load a trained model and test it.
    
    @input model_path: Path to the saved model
    @type model_path: str
    @input genre_path: Path to genre label file
    @type genre_path: str
    @input midi_path: Path to MIDI files
    @type midi_path: str
    @input batch_size: Batch size for testing
    @type batch_size: int
    
    @return: Test accuracy
    @rtype: float
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint['model_type']
    feature_dim = checkpoint['feature_dim']
    piano_roll_shape = checkpoint['piano_roll_shape']
    num_classes = checkpoint['num_classes']
    
    print(f"Model type: {model_type}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    if piano_roll_shape:
        print(f"Piano roll shape: {piano_roll_shape}")
    
    # Prepare dataset
    include_piano_roll = model_type != 'handcrafted'
    print(f"\nPreparing test dataset (include_piano_roll={include_piano_roll})...")
    dataset = prepare_dataset(
        genre_path="msd_tagtraum_cd1.cls",
        midi_path="lmd_matched",
        handcrafted_data_path="handcrafted_features.npz",
        piano_roll_data_path="piano_rolls.npz"
    )
    
    # Create test dataset
    if include_piano_roll:
        test_dataset = MIDIDataset(dataset['test_features'],
                                  dataset['test_labels'],
                                  dataset['test_piano_rolls'])
    else:
        test_dataset = MIDIDataset(dataset['test_features'],
                                  dataset['test_labels'])
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and load model
    model = get_model(model_type,
                     feature_dim=feature_dim,
                     piano_roll_shape=piano_roll_shape,
                     num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Test the model
    print("\nTesting model...")
    test_acc = test_model(model, test_loader, device, model_type)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_acc


def predict_single_midi(model_path, midi_file_path):
    """
    Predict the genre of a single MIDI file.
    
    @input model_path: Path to the saved model
    @type model_path: str
    @input midi_file_path: Path to the MIDI file to classify
    @type midi_file_path: str
    
    @return: Predicted genre
    @rtype: str
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint['model_type']
    feature_dim = checkpoint['feature_dim']
    piano_roll_shape = checkpoint['piano_roll_shape']
    num_classes = checkpoint['num_classes']
    label_list = checkpoint['label_list']
    
    # Create and load model
    model = get_model(model_type,
                     feature_dim=feature_dim,
                     piano_roll_shape=piano_roll_shape,
                     num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract features
    print(f"Extracting features from {midi_file_path}...")
    features = get_features(midi_file_path)
    
    if features is None:
        print("Error: Could not extract features from MIDI file.")
        return None
    
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        if model_type == 'handcrafted':
            outputs = model(features_tensor)
        else:
            piano_roll = get_piano_roll(midi_file_path)
            if piano_roll is None:
                print("Error: Could not extract piano roll from MIDI file.")
                return None
            piano_roll_tensor = torch.FloatTensor(piano_roll).unsqueeze(0).to(device)
            
            if model_type == 'piano_roll':
                outputs = model(piano_roll_tensor)
            else:  # combined models
                outputs = model(features_tensor, piano_roll_tensor)
        
        _, predicted = torch.max(outputs.data, 1)
        predicted_genre = label_list[predicted.item()]
    
    print(f"Predicted genre: {predicted_genre}")
    return predicted_genre


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test MIDI Genre Classification Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'predict'],
                       help='Mode: test on test set or predict single file')
    parser.add_argument('--midi_file', type=str, default=None,
                       help='Path to MIDI file for prediction (only for predict mode)')
    parser.add_argument('--genre_path', type=str, default='msd_tagtraum_cd1.cls',
                       help='Path to genre label file (only for test mode)')
    parser.add_argument('--midi_path', type=str, default='lmd_matched',
                       help='Path to MIDI files directory (only for test mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing (only for test mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_acc = load_and_test(
            model_path=args.model_path,
            genre_path=args.genre_path,
            midi_path=args.midi_path,
            batch_size=args.batch_size
        )
    elif args.mode == 'predict':
        if args.midi_file is None:
            print("Error: --midi_file must be specified for predict mode")
        else:
            predicted_genre = predict_single_midi(args.model_path, args.midi_file)
