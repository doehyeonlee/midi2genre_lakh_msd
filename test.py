import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import psutil

from model import get_model
from dataset_process import prepare_dataset, get_features, get_piano_roll
from train import MIDIDataset, validate


def test_model(model, dataloader, device, model_type):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if model_type == 'handcrafted':
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
            elif model_type == 'piano_roll':
                features, piano_rolls, labels = batch_data
                features = features.to(device)
                piano_rolls = piano_rolls.to(device)
                labels = labels.to(device)
                outputs = model(piano_rolls)
            else:
                features, piano_rolls, labels = batch_data
                features = features.to(device)
                piano_rolls = piano_rolls.to(device)
                labels = labels.to(device)
                outputs = model(features, piano_rolls)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_labels


def load_and_test(model_path,
                 genre_path="msd_tagtraum_cd1.cls",
                 midi_path="lmd_matched",
                 batch_size=32,
                 max_length=1000):
    """
    Load a trained model and test it on test set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Load checkpoint
    print(f"\n Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint['model_type']
    print(model_type)
    feature_dim = checkpoint['feature_dim']
    piano_roll_shape = checkpoint['piano_roll_shape']
    num_classes = checkpoint['num_classes']
    label_list = checkpoint.get('label_list', None)
    
    print(f"   Model type: {model_type}")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Number of classes: {num_classes}")
    if piano_roll_shape:
        print(f"   Piano roll shape: {piano_roll_shape}")
    
    # Prepare dataset
    print(f"\n📊 Loading test dataset...")
    dataset = prepare_dataset(
        genre_path=genre_path,
        midi_path=midi_path,
        handcrafted_data_path="processed_handcrafted_features.npz",
        piano_roll_base_path="processed_piano_rolls"
    )
    
    test_features = dataset['test_features']
    test_labels = dataset['test_labels']
    test_paths = dataset['test_paths']
    
    include_piano_roll = model_type != 'handcrafted'
    
    if include_piano_roll:
        test_chunk_files = dataset['piano_roll_metadata']['test_chunk_files']
        test_dataset = MIDIDataset(
            test_features,
            test_labels,
            test_paths,
            test_chunk_files,
            max_length=max_length
        )
    else:
        test_dataset = MIDIDataset(test_features, test_labels, test_paths)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create and load model
    
    print(f"\n Loading model weights...")
    model = get_model(
        model_type,
        feature_dim=feature_dim,
        piano_roll_shape=piano_roll_shape,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Test the model
    print("\n Testing model...")
    test_acc, predictions, true_labels = test_model(model, test_loader, device, model_type)
    
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    return test_acc, predictions, true_labels


def predict_single_midi(model_path, midi_file_path, max_length=1000):
    """
    Predict the genre of a single MIDI file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f" Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint['model_type']
    feature_dim = checkpoint['feature_dim']
    piano_roll_shape = checkpoint['piano_roll_shape']
    num_classes = checkpoint['num_classes']
    label_list = checkpoint['label_list']
    
    print(f"   Model type: {model_type}")
    
    # Create and load model
    model = get_model(
        model_type,
        feature_dim=feature_dim,
        piano_roll_shape=piano_roll_shape,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract features
    print(f"\n Extracting features from {midi_file_path}...")
    features = get_features(midi_file_path)
    
    if features is None:
        print(" Error: Could not extract features from MIDI file.")
        return None
    
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        if model_type == 'handcrafted':
            outputs = model(features_tensor)
        else:
            piano_roll = get_piano_roll(midi_file_path)
            if piano_roll is None:
                print("  Warning: Could not extract piano roll, using features only")
                outputs = model(features_tensor, None)  # Optional piano roll
            else:
                if len(piano_roll.shape) == 2:
                    current_length = piano_roll.shape[1]
                    if current_length < max_length:
                        piano_roll = np.pad(piano_roll, ((0, 0), (0, max_length - current_length)), mode='constant')
                    elif current_length > max_length:
                        piano_roll = piano_roll[:, :max_length]
                
                piano_roll_tensor = torch.FloatTensor(piano_roll).unsqueeze(0).to(device)
                outputs = model(features_tensor, piano_roll_tensor)
        
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_genre = label_list[predicted.item()]
    
    print(f"\n{'='*60}")
    print(f" Predicted genre: {predicted_genre}")
    print(f"   Confidence: {confidence.item()*100:.2f}%")
    print(f"{'='*60}")
    
    # Top 3 predictions
    print(f"\nTop 3 predictions:")
    top3_prob, top3_idx = torch.topk(probabilities[0], 3)
    for i, (prob, idx) in enumerate(zip(top3_prob, top3_idx)):
        print(f"   {i+1}. {label_list[idx.item()]}: {prob.item()*100:.2f}%")
    
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
    parser.add_argument('--max_length', type=int, default=1000,
                       help='Maximum length for piano rolls')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_acc, predictions, true_labels = load_and_test(
            model_path=args.model_path,
            genre_path=args.genre_path,
            midi_path=args.midi_path,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    elif args.mode == 'predict':
        if args.midi_file is None:
            print(" Error: --midi_file must be specified for predict mode")
        else:
            predicted_genre = predict_single_midi(
                args.model_path,
                args.midi_file,
                max_length=args.max_length
            )
