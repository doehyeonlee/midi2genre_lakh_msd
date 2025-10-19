import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os

from model import get_model
from dataset_process import prepare_dataset


class MIDIDataset(Dataset):
    def __init__(self, features, labels, piano_rolls=None, max_length=1000):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
        if piano_rolls is not None:
            self.piano_rolls = piano_rolls
            self.has_piano_rolls = True
        else:
            self.piano_rolls = None
            self.has_piano_rolls = False
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.has_piano_rolls:
            piano_roll = self.piano_rolls[idx]
            
            # Type 검증
            if not isinstance(piano_roll, np.ndarray):
                raise TypeError(f"Sample {idx}: Expected numpy array, got {type(piano_roll)}")
            
            # Shape 검증
            if len(piano_roll.shape) != 2:
                raise ValueError(f"Sample {idx}: Expected 2D array, got shape {piano_roll.shape}")
            
            current_pitches, current_length = piano_roll.shape
            
            # 1️⃣ Pitch 차원 조정 (128로 맞추기)
            if current_pitches < 128:
                pitch_pad = 128 - current_pitches
                piano_roll = np.pad(piano_roll, ((0, pitch_pad), (0, 0)), mode='constant')
            elif current_pitches > 128:
                piano_roll = piano_roll[:128, :]
            
            # 2️⃣ Time 차원 조정 (max_length로 맞추기)
            current_length = piano_roll.shape[1]
            
            if current_length < self.max_length:
                time_pad = self.max_length - current_length
                piano_roll = np.pad(piano_roll, ((0, 0), (0, time_pad)), mode='constant')
            elif current_length > self.max_length:
                piano_roll = piano_roll[:, :self.max_length]
            
            # 최종 검증
            assert piano_roll.shape == (128, self.max_length), \
                f"Sample {idx}: Final shape {piano_roll.shape} != expected (128, {self.max_length})"
            
            piano_roll_tensor = torch.FloatTensor(piano_roll)
            return feature, piano_roll_tensor, label
        else:
            return feature, label


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_data in dataloader:
        if model_type == 'handcrafted':
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
        else:
            features, piano_rolls, labels = batch_data
            features = features.to(device)
            piano_rolls = piano_rolls.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            if model_type == 'piano_roll':
                outputs = model(piano_rolls)
            else:  # combined models
                outputs = model(features, piano_rolls)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1, keepdim=True)
        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, model_type):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
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
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model_type='handcrafted', 
                genre_path="msd_tagtraum_cd1.cls",
                midi_path="lmd_matched",
                batch_size=32,
                num_epochs=50,
                learning_rate=0.001,
                save_path='models',
                max_length=1000):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = prepare_dataset(
        genre_path=genre_path,
        midi_path=midi_path,
        handcrafted_data_path="handcrafted_features.npz",
        piano_roll_data_path="piano_rolls.npz"
    )

    # Create datasets
    include_piano_roll = model_type != 'handcrafted'
    
    if include_piano_roll:
        train_dataset = MIDIDataset(
            dataset['train_features'], 
            dataset['train_labels'],
            dataset['train_piano_rolls'],
            max_length=max_length
        )
        val_dataset = MIDIDataset(
            dataset['val_features'],
            dataset['val_labels'],
            dataset['val_piano_rolls'],
            max_length=max_length
        )
    else:
        train_dataset = MIDIDataset(
            dataset['train_features'], 
            dataset['train_labels']
        )
        val_dataset = MIDIDataset(
            dataset['val_features'],
            dataset['val_labels']
        )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    feature_dim = dataset['train_features'].shape[1]
    piano_roll_shape = (128, max_length) if include_piano_roll else None
    num_classes = dataset['num_classes']
    
    print(f"\n🤖 Creating {model_type} model...")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Number of classes: {num_classes}")
    if piano_roll_shape:
        print(f"   Piano roll shape: {piano_roll_shape}")
    
    model = get_model(
        model_type, 
        feature_dim=feature_dim,
        piano_roll_shape=piano_roll_shape,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_loss, val_acc = validate(model, val_loader, criterion, device, model_type)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f'{model_type}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_dict': dataset['label_dict'],
                'label_list': dataset['label_list'],
                'num_classes': num_classes,
                'feature_dim': feature_dim,
                'piano_roll_shape': piano_roll_shape,
                'model_type': model_type
            }, save_file)
            print(f"  -> Saved best model with val_acc: {val_acc:.2f}%")
    
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MIDI Genre Classification Model')
    parser.add_argument('--model_type', type=str, default='handcrafted',
                       choices=['handcrafted', 'piano_roll', 'combined_linear', 'combined_cnn', 'combined_rnn'],
                       help='Type of model to train')
    parser.add_argument('--genre_path', type=str, default='msd_tagtraum_cd1.cls',
                       help='Path to genre label file')
    parser.add_argument('--midi_path', type=str, default='lmd_matched',
                       help='Path to MIDI files directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--max_length', type=int, default=1000,
                       help='Maximum length for piano rolls')
    
    args = parser.parse_args()
    
    model, history = train_model(
        model_type=args.model_type,
        genre_path=args.genre_path,
        midi_path=args.midi_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        max_length=args.max_length
    )
