import numpy as np
import pandas as pd
import pretty_midi
import warnings
import os

def get_genres(path):
    ids = []
    genres = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                [x, y, *_] = line.strip().split("\t")
                ids.append(x)
                genres.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
    return genre_df

def get_matched_midi(midi_folder, genre_df):
    # Get All Midi Files
    track_ids, file_paths = [], []
    for dir_name, subdir_list, file_list in os.walk(midi_folder):
        if len(dir_name) == 36:
            track_id = dir_name[18:]
            file_path_list = ["/".join([dir_name, file]) for file in file_list]
            for file_path in file_path_list:
                track_ids.append(track_id)
                file_paths.append(file_path)
    all_midi_df = pd.DataFrame({"TrackID": track_ids, "Path": file_paths})
    
    # Inner Join with Genre Dataframe
    df = pd.merge(all_midi_df, genre_df, on='TrackID', how='inner')
    return df.drop(["TrackID"], axis=1)

def normalize_features(features):
    # Normalizes the features to the range [-1, 1]
    tempo = (features[0] - 150) / 300
    resolution = (features[2] - 260) / 400
    time_sig_1 = (features[3] - 3) / 8
    time_sig_2 = (features[4] - 3) / 8
    return [tempo, resolution, time_sig_1, time_sig_2]

def get_features(path):
    try:
        # Test for Corrupted Midi Files
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            file = pretty_midi.PrettyMIDI(path)
            
            tempo = file.estimate_tempo()
            num_sig_changes = len(file.time_signature_changes)
            resolution = file.resolution
            ts_changes = file.time_signature_changes
            ts_1 = 4
            ts_2 = 4
            if len(ts_changes) > 0:
                ts_1 = ts_changes[0].numerator
                ts_2 = ts_changes[0].denominator
            return normalize_features([tempo, num_sig_changes, resolution, ts_1, ts_2])
    except:
        return None

def get_piano_roll(path, fs=100):
    """
    Extract piano roll representation from MIDI file.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            file = pretty_midi.PrettyMIDI(path)
            piano_roll = file.get_piano_roll(fs=fs)
            if piano_roll.max() > 0:
                piano_roll = piano_roll / piano_roll.max()
            return piano_roll
    except:
        return None

def save_features(indices, label_list, label_dict, labels_array, features_array, paths_array,
                 handcrafted_data_path="processed_handcrafted_features.npz"):
    features = features_array[indices]
    labels = labels_array[indices]
    paths = paths_array[indices]

    num = len(labels)
    num_training = int(num * 0.8)  # 80% train
    
    print(f"   Train: {num_training} samples (80%)")
    print(f"   Test: {num - num_training} samples (20%)")
    
    handcrafted_dataset = {
        'train_features': features[:num_training],
        'train_labels': labels[:num_training],
        'train_paths': paths[:num_training],
        'test_features': features[num_training:],
        'test_labels': labels[num_training:],
        'test_paths': paths[num_training:],
        'label_list': label_list,
        'label_dict': label_dict,
        'num_classes': len(label_list)
    }
    
    print(f"Saving handcrafted features to '{handcrafted_data_path}'...")
    np.savez_compressed(handcrafted_data_path, **handcrafted_dataset)
    return handcrafted_dataset

def save_piano_rolls_in_chunks(train_paths, test_paths, train_labels, test_labels,
                               piano_roll_base_path="processed_piano_rolls",
                               chunk_size=1000, fs=100):
    """
    Piano rolls를 chunk 단위(기본 1000개)로 나눠서 개별 파일로 저장
    """
    
    def extract_and_save_chunks(paths, labels, split_name, base_path):
        num_chunks = (len(paths) + chunk_size - 1) // chunk_size
        chunk_files = []
        
        print(f"\nProcessing {split_name} set ({len(paths)} samples, {num_chunks} chunks)...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(paths))
            chunk_paths = paths[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            chunk_piano_rolls = []
            valid_indices = []
            
            print(f"\n  Chunk {chunk_idx + 1}/{num_chunks} ({start_idx}-{end_idx}):")
            for i, path in enumerate(chunk_paths):
                if i % 100 == 0 and i > 0:
                    print(f"    Progress: {i}/{len(chunk_paths)} ({100*i/len(chunk_paths):.1f}%)")
                
                piano_roll = get_piano_roll(path, fs=fs)
                if piano_roll is not None and piano_roll.size > 0:
                    chunk_piano_rolls.append(piano_roll)
                    valid_indices.append(i)
            
            piano_rolls_array = np.empty(len(chunk_piano_rolls), dtype=object)
            for i, pr in enumerate(chunk_piano_rolls):
                piano_rolls_array[i] = pr
            
            valid_labels = chunk_labels[valid_indices]
            valid_paths = chunk_paths[valid_indices]
            
            # 청크 파일로 저장
            chunk_file = f"{base_path}_{split_name}_chunk_{chunk_idx:04d}.npz"
            try:
                np.savez_compressed(chunk_file,
                                   piano_rolls=piano_rolls_array,
                                   labels=valid_labels,
                                   paths=valid_paths)
                file_size = os.path.getsize(chunk_file) / (1024 * 1024)
                print(f"    ✓ Saved {len(chunk_piano_rolls)}/{len(chunk_paths)} samples to {chunk_file} ({file_size:.2f} MB)")
                chunk_files.append(chunk_file)
            except Exception as e:
                print(f"    ✗ Failed to save chunk {chunk_idx}: {e}")
        
        return chunk_files
    
    # Train과 Test 각각 청크로 저장
    train_files = extract_and_save_chunks(train_paths, train_labels, "train", piano_roll_base_path)
    test_files = extract_and_save_chunks(test_paths, test_labels, "test", piano_roll_base_path)
    
    # 메타데이터 저장 (어떤 청크 파일들이 있는지)
    metadata = {
        'train_chunk_files': train_files,
        'test_chunk_files': test_files,
        'chunk_size': chunk_size
    }
    
    metadata_file = f"{piano_roll_base_path}_metadata.npz"
    np.savez_compressed(metadata_file, **metadata)
    print(f"\n✓ Metadata saved to {metadata_file}")
    print(f"  Train: {len(train_files)} chunks")
    print(f"  Test: {len(test_files)} chunks")
    
    return metadata

def extract_features_only(path_df, label_dict, random_seed=42):
    """
    Handcrafted features만 먼저 추출 (piano roll 제외)
    """
    all_features = []
    all_labels = []
    all_paths = []

    for index, row in path_df.iterrows():
        if index % 100 == 0 and index > 0:
            print(f"  Progress: {index}/{len(path_df)} ({100*index/len(path_df):.1f}%)")

        features = get_features(row.Path)
        genre = label_dict[row.Genre]
        
        if features is not None and type(genre) == int:
            all_features.append(features)
            all_labels.append(genre)
            all_paths.append(row.Path)
    
    print(f"Successfully extracted {len(all_labels)} samples out of {len(path_df)}")
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    paths_array = np.array(all_paths)

    return {
        'features': features_array,
        'labels': labels_array,
        'paths': paths_array
    }


def prepare_dataset(genre_path="msd_tagtraum_cd1.cls", 
                   midi_path="lmd_matched",
                   handcrafted_data_path="processed_handcrafted_features.npz",
                   piano_roll_base_path="processed_piano_rolls",
                   fs=100,
                   piano_roll_chunk_size=1000,
                   random_seed=42):

    if os.path.exists(handcrafted_data_path) and os.path.exists(f"{piano_roll_base_path}_metadata.npz"):
        print("Loading existing datasets...")
        loaded_data = np.load(handcrafted_data_path, allow_pickle=True)
        dataset = {}
        for key in loaded_data.files:
            if key == 'label_dict':
                dataset[key] = loaded_data[key].item() 
            elif key == 'label_list':
                dataset[key] = loaded_data[key].tolist()
            else:
                dataset[key] = loaded_data[key]

        metadata = np.load(f"{piano_roll_base_path}_metadata.npz", allow_pickle=True)
        dataset['piano_roll_metadata'] = {
            'train_chunk_files': metadata['train_chunk_files'].tolist(),
            'test_chunk_files': metadata['test_chunk_files'].tolist(),
            'chunk_size': int(metadata['chunk_size'])
        }
        
        print("Datasets loaded successfully.")
        print(f"  Train chunks: {len(dataset['piano_roll_metadata']['train_chunk_files'])}")
        print(f"  Test chunks: {len(dataset['piano_roll_metadata']['test_chunk_files'])}")
        return dataset

    else:
        genre_df = get_genres(genre_path)
        label_list = list(set(genre_df.Genre))
        label_dict = {lbl: label_list.index(lbl) for lbl in label_list}
        
        print(f"\nGenres ({len(label_list)}): {label_list}\n")
        
        matched_midi_df = get_matched_midi(midi_path, genre_df)
        print(f"Found {len(matched_midi_df)} matched MIDI files\n")

        # Step 1: Handcrafted features 먼저 추출 및 저장
        print("="*60)
        print("STEP 1: Extracting Handcrafted Features")
        print("="*60)
        all_data = extract_features_only(
            path_df=matched_midi_df,
            label_dict=label_dict,
            random_seed=random_seed
        )
        
        np.random.seed(random_seed)
        num_samples = len(all_data['labels'])
        indices = np.random.permutation(num_samples)
        
        print(f"\nSaving handcrafted features...")
        handcrafted_dataset = save_features(
            indices=indices,
            label_list=label_list,
            label_dict=label_dict,
            labels_array=all_data['labels'],
            features_array=all_data['features'],
            paths_array=all_data['paths'],
            handcrafted_data_path=handcrafted_data_path
        )
        
        # Step 2: Piano rolls를 chunk 단위로 추출 및 저장
        print("\n" + "="*60)
        print("STEP 2: Extracting Piano Rolls (in chunks)")
        print("="*60)
        metadata = save_piano_rolls_in_chunks(
            train_paths=handcrafted_dataset['train_paths'],
            test_paths=handcrafted_dataset['test_paths'],
            train_labels=handcrafted_dataset['train_labels'],
            test_labels=handcrafted_dataset['test_labels'],
            piano_roll_base_path="processed_piano_rolls",
            chunk_size=piano_roll_chunk_size,
            fs=fs
        )

        dataset = {
            'train_features': handcrafted_dataset['train_features'],
            'test_features': handcrafted_dataset['test_features'],
            'train_labels': handcrafted_dataset['train_labels'],
            'test_labels': handcrafted_dataset['test_labels'],
            'train_paths': handcrafted_dataset['train_paths'],
            'test_paths': handcrafted_dataset['test_paths'],
            'piano_roll_metadata': metadata,
            'label_list': label_list,
            'label_dict': label_dict,
            'num_classes': len(label_list)
        }
        return dataset


if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)
    
    print("="*60)
    print("MIDI Genre Classification - Data Preprocessing")
    print("="*60)
    dataset = prepare_dataset(
        genre_path="msd_tagtraum_cd1.cls",
        midi_path="lmd_matched",
        handcrafted_data_path="processed_handcrafted_features.npz",
        piano_roll_base_path="processed_piano_rolls",
        fs=100,
        piano_roll_chunk_size=1000,
        random_seed=random_seed
    )

    print("\n" + "="*60)
    print("데이터셋 정보")
    print("="*60)
    print(f"Train samples: {len(dataset['train_labels'])}")
    print(f"Test samples: {len(dataset['test_labels'])}")
    print(f"Number of classes: {dataset['num_classes']}")
    print(f"Feature dimension: {dataset['train_features'].shape[1]}")
    if 'piano_roll_metadata' in dataset:
        print(f"Piano roll chunks:")
        print(f"  Train: {len(dataset['piano_roll_metadata']['train_chunk_files'])} files")
        print(f"  Test: {len(dataset['piano_roll_metadata']['test_chunk_files'])} files")
    print("="*60)
