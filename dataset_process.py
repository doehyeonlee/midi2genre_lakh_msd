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

def save_features(indices, label_list, label_dict, labels_array, features_array, piano_rolls_array,
                 handcrafted_data_path="processed_handcrafted_features.npz",
                 piano_roll_data_path="processed_piano_rolls.npz"):

    features = features_array[indices]
    piano_rolls = piano_rolls_array[indices]
    labels = labels_array[indices]

    num = len(labels)
    num_training = int(num * 0.6)
    num_validation = int(num * 0.8)
    
    print(f"   Train: {num_training} samples (60%)")
    print(f"   Val: {num_validation - num_training} samples (20%)")
    print(f"   Test: {num - num_validation} samples (20%)")
    
    handcrafted_dataset = {
        'train_features': features[:num_training],
        'train_labels': labels[:num_training],
        'val_features': features[num_training:num_validation],
        'val_labels': labels[num_training:num_validation],
        'test_features': features[num_validation:],
        'test_labels': labels[num_validation:],
        'label_list': label_list,
        'label_dict': label_dict,
        'num_classes': len(label_list)
    }
    
    piano_roll_dataset = {
        'train_piano_rolls': piano_rolls[:num_training],
        'train_labels': labels[:num_training],
        'val_piano_rolls': piano_rolls[num_training:num_validation],
        'val_labels': labels[num_training:num_validation],
        'test_piano_rolls': piano_rolls[num_validation:],
        'test_labels': labels[num_validation:]
    }
    
    print(f"Saving handcrafted features to '{handcrafted_data_path}'...")
    np.savez_compressed(handcrafted_data_path, **handcrafted_dataset)
        
    print(f"Saving piano rolls to '{piano_roll_data_path}'...")
    np.savez_compressed(piano_roll_data_path, **piano_roll_dataset)

    return handcrafted_dataset, piano_roll_dataset, labels

def extract_features(path_df, label_dict, fs=100,
                    temp_handcrafted_path="temp_handcrafted_features.npz",
                    temp_piano_roll_path="temp_piano_rolls.npz",
                    label_list=None, random_seed=42):

    all_features = []
    all_piano_rolls = []
    all_labels = []
    all_paths = []
    
    print(f"Extracting data from {len(path_df)} MIDI files...")
    print("Processing Dataset from scratch...")

    for index, row in path_df.iterrows():
        if index % 100 == 0 and index > 0:
            print(f"  Progress: {index}/{len(path_df)} ({100*index/len(path_df):.1f}%)")
        
        if index % 5000 == 0 and index > 0:
            temp_features = np.array(all_features)
            temp_piano_rolls = np.empty(len(all_piano_rolls), dtype=object)
            for i, pr in enumerate(all_piano_rolls):
                temp_piano_rolls[i] = pr
            temp_labels = np.array(all_labels)
            np.random.seed(random_seed)
            temp_indices = np.random.permutation(len(temp_labels))
            
            save_features(
                indices=temp_indices,
                label_list=label_list,
                label_dict=label_dict,
                labels_array=temp_labels,
                features_array=temp_features,
                piano_rolls_array=temp_piano_rolls,
                handcrafted_data_path=temp_handcrafted_path,
                piano_roll_data_path=temp_piano_roll_path
            )
            print(f"Checkpoint saved!\n")

        features = get_features(row.Path)
        piano_roll = get_piano_roll(row.Path, fs=fs)
        genre = label_dict[row.Genre]
        
        if features is not None and piano_roll is not None and type(genre) == int:
            all_features.append(features)
            all_piano_rolls.append(piano_roll)
            all_labels.append(genre)
            all_paths.append(row.Path)
    
    print(f"Successfully extracted {len(all_labels)} samples out of {len(path_df)}")
    
    features_array = np.array(all_features)
    piano_rolls_array = np.empty(len(all_piano_rolls), dtype=object)
    for i, pr in enumerate(all_piano_rolls):
        piano_rolls_array[i] = pr
    labels_array = np.array(all_labels)

    lengths = [pr.shape[1] for pr in piano_rolls_array[:min(100, len(piano_rolls_array))]]
    print(f"Piano roll lengths (first 100): min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    return {
        'features': features_array,
        'piano_rolls': piano_rolls_array,
        'labels': labels_array,
        'paths': all_paths
    }


def prepare_dataset(genre_path="msd_tagtraum_cd1.cls", 
                   midi_path="lmd_matched",
                   handcrafted_data_path="temp_handcrafted_features.npz",
                   piano_roll_data_path="temp_piano_rolls.npz",
                   fs=100,
                   save_interval=1000,
                   random_seed=42):

    if os.path.exists(handcrafted_data_path) and os.path.exists(piano_roll_data_path):
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

        #piano_data = np.load(piano_roll_data_path, allow_pickle=True)
            
        #dataset['train_piano_rolls'] = piano_data['train_piano_rolls']
        #dataset['val_piano_rolls'] = piano_data['val_piano_rolls']
        #dataset['test_piano_rolls'] = piano_data['test_piano_rolls']
        print("Datasets loaded successfully.")
        return dataset

    else:
        genre_df = get_genres(genre_path)
        label_list = list(set(genre_df.Genre))
        label_dict = {lbl: label_list.index(lbl) for lbl in label_list}
        
        print(f"\nGenres ({len(label_list)}): {label_list}\n")
        
        matched_midi_df = get_matched_midi(midi_path, genre_df)
        print(f"Found {len(matched_midi_df)} matched MIDI files\n")

        all_data = extract_features(
            path_df=matched_midi_df,
            label_dict=label_dict,
            fs=fs,
            temp_handcrafted_path="temp_handcrafted_features.npz",
            temp_piano_roll_path="temp_piano_rolls.npz",
            label_list=label_list,
            random_seed=random_seed
        )
        np.random.seed(random_seed)
        num_samples = len(all_data['labels'])
        indices = np.random.permutation(num_samples)
        print(f"Saving final datasets...")
        handcrafted_dataset, piano_roll_dataset, labels = save_features(
            indices=indices,
            label_list=label_list,
            label_dict=label_dict,
            labels_array=all_data['labels'],
            features_array=all_data['features'],
            piano_rolls_array=all_data['piano_rolls'],
            handcrafted_data_path=handcrafted_data_path,
            piano_roll_data_path=piano_roll_data_path
        )

        dataset = {
            'train_features': handcrafted_dataset['train_features'],
            'val_features': handcrafted_dataset['val_features'],
            'test_features': handcrafted_dataset['test_features'],
            'train_piano_rolls': piano_roll_dataset['train_piano_rolls'],
            'val_piano_rolls': piano_roll_dataset['val_piano_rolls'],
            'test_piano_rolls': piano_roll_dataset['test_piano_rolls'],
            'train_labels': labels['train_labels'],
            'val_labels': labels['val_labels'],
            'test_labels': labels['test_labels'],
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
        piano_roll_data_path="processed_piano_rolls.npz",
        fs=100,
        random_seed=random_seed
    )

    print("\n" + "="*60)
    print("데이터셋 정보 (Handcrafted Features Only)")
    print("="*60)
    print(f"Train samples: {len(dataset['train_labels'])}")
    print(f"Val samples: {len(dataset['val_labels'])}")
    print(f"Test samples: {len(dataset['test_labels'])}")
    print(f"Number of classes: {dataset['num_classes']}")
    print(f"Feature dimension: {dataset['train_features'].shape[1]}")
    print("="*60)
