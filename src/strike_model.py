import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):  #INCLUDE IN IMPLEMENTATION THIS IS SICK
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1) #projects hidden states to attention scores
    
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1) #normalise to sum to 1
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class StrikeNet(nn.Module):
    def __init__(self, num_classes=8, sequence_length=15, use_attention=True, dropout_rate=0.5):
        super(StrikeNet, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        
        self.frame_features = 17 * 3 #17 keypoints with x,y,conf
        
        self.fc_frame1 = nn.Linear(self.frame_features, 128)
        self.fc_frame2 = nn.Linear(128, 256)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fighter_embedding = nn.Embedding(2, 16) #fighter-specific features
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True #capture past and future context
        )
        
        if use_attention:
            self.attention = TemporalAttention(256 * 2)
        
        self.fc1 = nn.Linear(256 * 2 + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.fc_class = nn.Linear(64, num_classes) #strike classification head
        self.fc_outcome = nn.Linear(64, 2) #hit/miss classification head
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, fighter_id=None):
        batch_size, sequence_length, num_joints, channels = x.shape
        
        frame_features = []
        for t in range(sequence_length):
            frame = x[:, t, :, :]
            frame = frame.reshape(batch_size, -1) #flatten keypoints
            
            frame = F.relu(self.bn1(self.fc_frame1(frame)))
            frame = F.relu(self.bn2(self.fc_frame2(frame)))
            
            frame_features.append(frame)
        
        frame_features = torch.stack(frame_features, dim=1)
        
        lstm_out, _ = self.lstm(frame_features)
        
        if self.use_attention:
            context, _ = self.attention(lstm_out) #focus on key frames
        else:
            context = lstm_out[:, -1, :]
        
        if fighter_id is not None:
            fighter_embedding = self.fighter_embedding(fighter_id)
            context = torch.cat([context, fighter_embedding], dim=1)
        else:
            fighter_embedding = torch.zeros(batch_size, 16, device=context.device)
            context = torch.cat([context, fighter_embedding], dim=1)
        
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        class_logits = self.fc_class(x)
        outcome_logits = self.fc_outcome(x)
        
        return class_logits, outcome_logits