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
    def __init__(self, features, labels, paths, piano_roll_chunk_files=None, max_length=1000):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.paths = paths
        self.max_length = max_length
        
        if piano_roll_chunk_files is not None:
            self.has_piano_rolls = True
            self._cache_piano_rolls(piano_roll_chunk_files, set(paths))
        else:
            self.has_piano_rolls = False
            self.piano_roll_cache = None

    def __len__(self):
        return len(self.labels)
    
    def _cache_piano_rolls(self, chunk_files, target_paths_set):
        import psutil
        
        mem = psutil.virtual_memory()
        print(f"\n Memory status before caching:")
        print(f"   Available: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
        
        self.piano_roll_cache = {}
        total_loaded = 0
        
        print(f"\n Caching piano rolls from {len(chunk_files)} chunks...")
        
        for chunk_idx, chunk_file in enumerate(chunk_files):
            if not os.path.exists(chunk_file):
                print(f"   Warning: Chunk file not found: {chunk_file}")
                continue
            
            data = np.load(chunk_file, allow_pickle=True)
            chunk_paths = data['paths']
            chunk_piano_rolls = data['piano_rolls']
            
            matches = 0
            for path, pr in zip(chunk_paths, chunk_piano_rolls):
                if path in target_paths_set:
                    self.piano_roll_cache[path] = pr
                    matches += 1
            
            total_loaded += matches
            if matches > 0:
                print(f"   Chunk {chunk_idx+1}/{len(chunk_files)}: cached {matches} piano rolls")
        
        mem_after = psutil.virtual_memory()
        print(f"\n✓ Cached {total_loaded} piano rolls")
        print(f" Memory status after caching:")
        print(f"   Available: {mem_after.available / (1024**3):.2f} GB / {mem_after.total / (1024**3):.2f} GB")
        print(f"   Used for caching: ~{(mem.available - mem_after.available) / (1024**3):.2f} GB")
    

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.has_piano_rolls:
            path = self.paths[idx]
            piano_roll = self.piano_roll_cache.get(path, np.array([]))

            if piano_roll.size == 0:
                piano_roll = np.zeros((128, self.max_length), dtype=np.float32)
            else:
                if len(piano_roll.shape) != 2:
                    raise ValueError(f"Sample {idx}: Expected 2D array, got shape {piano_roll.shape}")
                
                current_length = piano_roll.shape[1]
                if current_length < self.max_length:
                    time_pad = self.max_length - current_length
                    piano_roll = np.pad(piano_roll, ((0, 0), (0, time_pad)), mode='constant')
                elif current_length > self.max_length:
                    piano_roll = piano_roll[:, :self.max_length]
            assert piano_roll.shape == (128, self.max_length), \
                f"Sample {idx}: Final shape {piano_roll.shape} != expected (128, {self.max_length})"
            
            piano_roll_tensor = torch.FloatTensor(piano_roll)
            return feature, piano_roll_tensor, label
        else:
            return feature, label


def balance_dataset(features, labels, paths, target_class=6, max_samples=500, random_seed=42):
    np.random.seed(random_seed)
    
    target_mask = labels == target_class
    target_indices = np.where(target_mask)[0]
    other_indices = np.where(~target_mask)[0]

    if len(target_indices) > max_samples:
        sampled_target = np.random.choice(target_indices, size=max_samples, replace=False)
        print(f"Undersampling Class {target_class}: {len(target_indices)} -> {max_samples} samples")
    else:
        sampled_target = target_indices
        print(f"Class {target_class}: {len(target_indices)} samples (no undersampling needed)")
    
    final_indices = np.concatenate([sampled_target, other_indices])
    np.random.shuffle(final_indices)
    
    balanced_features = features[final_indices]
    balanced_labels = labels[final_indices]
    balanced_paths = paths[final_indices]

    print("\nBalanced Class Distribution:")
    unique, counts = np.unique(balanced_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = 100 * count / len(balanced_labels)
        print(f"   Class {cls}: {count:4d} samples ({percentage:5.2f}%)")
    
    return balanced_features, balanced_labels, balanced_paths

def get_matched_piano_rolls(paths, chunk_files):
    all_chunk_paths = {}
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            data = np.load(chunk_file, allow_pickle=True)
            chunk_paths = data['paths']
            all_chunk_paths[chunk_file] = set(chunk_paths)
    
    paths_set = set(paths)
    matched_chunks = []
    
    for chunk_file, chunk_paths in all_chunk_paths.items():
        if paths_set & chunk_paths:  
            matched_chunks.append(chunk_file)
    
    return matched_chunks


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
                max_length=1000,
                balance_data=True,
                undersample_class=7,
                max_samples=500):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n Loading dataset...")
    dataset = prepare_dataset(
        genre_path=genre_path,
        midi_path=midi_path,
        handcrafted_data_path="processed_handcrafted_features.npz",
        piano_roll_base_path="processed_piano_rolls",
        fs=100,
        piano_roll_chunk_size=1000
    )
    
    # 원본 train set 가져오기 (80%)
    train_features_full = dataset['train_features']
    train_labels_full = dataset['train_labels']
    train_paths_full = dataset['train_paths']

    if balance_data:
        print(f"\n  Balancing training set by undersampling class {undersample_class}...")
        train_features_full, train_labels_full, train_paths_full = balance_dataset(
            train_features_full,
            train_labels_full,
            train_paths_full,
            target_class=undersample_class,
            max_samples=max_samples
        )
    else:
        print("\n Using original (imbalanced) training dataset...")

    print("\n  Splitting train set into train(90%) / validation(10%)...")
    np.random.seed(42)
    num_train_full = len(train_labels_full)
    indices = np.random.permutation(num_train_full)
    
    split_idx = int(num_train_full * 0.9)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_features = train_features_full[train_indices]
    train_labels = train_labels_full[train_indices]
    train_paths = train_paths_full[train_indices]
    
    val_features = train_features_full[val_indices]
    val_labels = train_labels_full[val_indices]
    val_paths = train_paths_full[val_indices]
    
    print(f"   Train: {len(train_labels)} samples (90%)")
    print(f"   Val: {len(val_labels)} samples (10%)")
    
    include_piano_roll = model_type != 'handcrafted'
    
    if include_piano_roll:
        print("\nLoading piano roll data...")
        train_chunk_files_full = dataset['piano_roll_metadata']['train_chunk_files']
        
        # Train/Val에 맞는 chunk만 선택
        train_chunk_files = get_matched_piano_rolls(train_paths, train_chunk_files_full)
        val_chunk_files = get_matched_piano_rolls(val_paths, train_chunk_files_full)
        
        train_dataset = MIDIDataset(
            train_features, 
            train_labels,
            train_paths,
            train_chunk_files,
            max_length=max_length
        )
        val_dataset = MIDIDataset(
            val_features,
            val_labels,
            val_paths,
            val_chunk_files,
            max_length=max_length
        )
    else:
        train_dataset = MIDIDataset(train_features, train_labels, train_paths)
        val_dataset = MIDIDataset(val_features, val_labels, val_paths)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    feature_dim = train_features.shape[1]
    piano_roll_shape = (128, max_length) if include_piano_roll else None
    num_classes = dataset['num_classes']
    
    print(f"\nCreating {model_type} model...")
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
    print("\n Using Cross Entropy Loss")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    print(f"\n Starting training for {num_epochs} epochs...")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_loss, val_acc = validate(model, val_loader, criterion, device, model_type)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.2f}%", end='')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
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
            print(f" ✓ Best!")
        else:
            patience_counter += 1
            print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
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
    parser.add_argument('--balance_data', action='store_true', default=True,
                       help='Balance dataset by undersampling majority class')
    parser.add_argument('--no_balance', dest='balance_data', action='store_false',
                       help='Use original imbalanced dataset')
    parser.add_argument('--undersample_class', type=int, default=6,
                       help='Class to undersample (default: 6)')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum samples for undersampled class (default: 500)')
    
    args = parser.parse_args()
    
    model, history = train_model(
        model_type=args.model_type,
        genre_path=args.genre_path,
        midi_path=args.midi_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        max_length=args.max_length,
        balance_data=args.balance_data,
        undersample_class=args.undersample_class,
        max_samples=args.max_samples
    )
