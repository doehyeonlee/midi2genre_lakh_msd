import torch
import torch.nn as nn
import torch.nn.functional as F


class HandcraftedFeatureModel(nn.Module):
    """
    Model 1: Simple MLP using only handcrafted features (4 features).
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_classes=13):
        super(HandcraftedFeatureModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PianoRollLinearModel(nn.Module):
    """
    Model 2: Linear model using only piano roll data (128 x max_length).
    """
    def __init__(self, piano_roll_shape=(128, 1000), num_classes=13):
        super(PianoRollLinearModel, self).__init__()
        input_dim = piano_roll_shape[0] * piano_roll_shape[1]
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, piano_roll):
        x = piano_roll.view(piano_roll.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CombinedLinearModel(nn.Module):
    """
    Model 3: Linear model combining both handcrafted features and piano roll.
    """
    def __init__(self, feature_dim=4, piano_roll_shape=(128, 1000), num_classes=13):
        super(CombinedLinearModel, self).__init__()
        piano_roll_dim = piano_roll_shape[0] * piano_roll_shape[1]
        
        self.piano_fc1 = nn.Linear(piano_roll_dim, 256)
        self.piano_fc2 = nn.Linear(256, 64)
        
        self.feature_fc = nn.Linear(feature_dim, 32)
        combined_dim = 64 + 32
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features, piano_roll):
        pr = piano_roll.view(piano_roll.size(0), -1)
        pr = F.relu(self.piano_fc1(pr))
        pr = self.dropout(pr)
        pr = F.relu(self.piano_fc2(pr))
        feat = F.relu(self.feature_fc(features))
        
        x = torch.cat([pr, feat], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CombinedCNNModel(nn.Module):
    """
    Model 4: CNN model combining both handcrafted features and piano roll.
    """
    def __init__(self, feature_dim=4, piano_roll_shape=(128, 1000), num_classes=13):
        super(CombinedCNNModel, self).__init__()
        
        # CNN for piano roll (treat as 1-channel image)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after convolutions
        # (128, 1000) -> (64, 500) -> (32, 250) -> (16, 125)
        cnn_output_dim = 64 * 16 * 125
        
        self.piano_fc = nn.Linear(cnn_output_dim, 128)
        self.feature_fc = nn.Linear(feature_dim, 32)
        
        combined_dim = 128 + 32
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features, piano_roll):
        pr = piano_roll.unsqueeze(1)
        pr = self.pool1(F.relu(self.conv1(pr)))
        pr = self.pool2(F.relu(self.conv2(pr)))
        pr = self.pool3(F.relu(self.conv3(pr)))
        
        pr = pr.view(pr.size(0), -1)
        pr = F.relu(self.piano_fc(pr))
        pr = self.dropout(pr)
        
        feat = F.relu(self.feature_fc(features))
        
        x = torch.cat([pr, feat], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CombinedRNNModel(nn.Module):
    """
    Model 5: RNN model combining both handcrafted features and piano roll.
    Piano roll is treated as a sequence (time steps x pitches).
    """
    def __init__(self, feature_dim=4, piano_roll_shape=(128, 1000), num_classes=13, hidden_dim=128):
        super(CombinedRNNModel, self).__init__()
        
        # LSTM for piano roll sequence
        # Input: (batch, seq_len=1000, input_size=128)
        self.lstm = nn.LSTM(input_size=piano_roll_shape[0], 
                           hidden_size=hidden_dim, 
                           num_layers=2, 
                           batch_first=True,
                           dropout=0.2)
        
        self.piano_fc = nn.Linear(hidden_dim, 64)
        self.feature_fc = nn.Linear(feature_dim, 32)
        # Combined processing
        combined_dim = 64 + 32
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features, piano_roll):
        # Transpose: (batch, 128, 1000) -> (batch, 1000, 128)
        pr = piano_roll.transpose(1, 2)
        
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(pr)
        
        pr = hidden[-1]  # (batch, hidden_dim)
        pr = F.relu(self.piano_fc(pr))
        pr = self.dropout(pr)
        
        feat = F.relu(self.feature_fc(features))
        
        x = torch.cat([pr, feat], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_type, feature_dim=4, piano_roll_shape=(128, 1000), num_classes=13):
    if model_type == 'handcrafted':
        return HandcraftedFeatureModel(input_dim=feature_dim, num_classes=num_classes)
    elif model_type == 'piano_roll':
        return PianoRollLinearModel(piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    elif model_type == 'combined_linear':
        return CombinedLinearModel(feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    elif model_type == 'combined_cnn':
        return CombinedCNNModel(feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    elif model_type == 'combined_rnn':
        return CombinedRNNModel(feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    batch_size = 4
    feature_dim = 4
    piano_roll_shape = (128, 1000)
    num_classes = 13
    
    # Create dummy data
    features = torch.randn(batch_size, feature_dim)
    piano_roll = torch.randn(batch_size, piano_roll_shape[0], piano_roll_shape[1])
    
    print("Testing models...")
    
    # Test Model 1
    model1 = get_model('handcrafted', feature_dim=feature_dim, num_classes=num_classes)
    out1 = model1(features)
    print(f"HandcraftedFeatureModel output shape: {out1.shape}")
    
    # Test Model 2
    model2 = get_model('piano_roll', piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    out2 = model2(piano_roll)
    print(f"PianoRollLinearModel output shape: {out2.shape}")
    
    # Test Model 3
    model3 = get_model('combined_linear', feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    out3 = model3(features, piano_roll)
    print(f"CombinedLinearModel output shape: {out3.shape}")
    
    # Test Model 4
    model4 = get_model('combined_cnn', feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    out4 = model4(features, piano_roll)
    print(f"CombinedCNNModel output shape: {out4.shape}")
    
    # Test Model 5
    model5 = get_model('combined_rnn', feature_dim=feature_dim, piano_roll_shape=piano_roll_shape, num_classes=num_classes)
    out5 = model5(features, piano_roll)
    print(f"CombinedRNNModel output shape: {out5.shape}")
