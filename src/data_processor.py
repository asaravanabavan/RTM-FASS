import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

from src.pose_detector import PoseDetector

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

class StrikeDataProcessor:
    def __init__(self, csv_file_path, output_directory='data/strike_dataset', sequence_length=15):
        self.csv_path = csv_file_path
        self.output_directory = Path(output_directory)
        self.sequence_length = sequence_length
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.pose_detector = PoseDetector(
            model="yolov8x-pose.pt",
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            conf_thresh=0.25
        )
            
    def process_dataset(self):
        print("Processing dataset...")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: CSV file not found at {self.csv_path}")
            return None
        
        dataframe = pd.read_csv(self.csv_path) #load padwork annotation csv file
        print(f"Labels in dataset: {dict(dataframe['label'].value_counts())}")
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        sequences = []
        labels = []
        outcomes = []
        
        import numpy as np
        
        valid_count = 0
        
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing videos"):
            clip_path = row['out']
            
            if not os.path.exists(clip_path):
                print(f"Warning: File not found {clip_path}, skipping...")
                continue
            
            try:
                video_capture = cv2.VideoCapture(clip_path)
                if not video_capture.isOpened():
                    print(f"Warning: Could not open {clip_path}, skipping...")
                    continue
                
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                min_frames_needed = self.sequence_length
                if total_frames < min_frames_needed:
                    print(f"Warning: Clip too short ({total_frames} frames, need {min_frames_needed}), skipping...")
                    video_capture.release()
                    continue
                
                sample_indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int) #evenly distribute frames
                
                pose_sequence = []
                
                for frame_index in sample_indices:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    return_value, frame = video_capture.read()
                    
                    if not return_value:
                        print(f"Warning: Failed to read frame {frame_index}, skipping...")
                        break
                    
                    try:
                        _, people, _ = self.pose_detector.detect_pose(frame)
                        
                        if people and len(people) > 0 and people[0].landmarks is not None:
                            landmarks = people[0].landmarks
                            landmarks_list = []
                            for point in landmarks:
                                landmarks_list.append([float(point[0]), float(point[1]), float(point[2])])
                            pose_sequence.append(landmarks_list)
                        else:
                            empty_pose = [[0.0, 0.0, 0.0] for _ in range(17)] #handle missing poses
                            pose_sequence.append(empty_pose)
                            
                    except Exception as pose_error:
                        print(f"Pose detection error on frame {frame_index}: {pose_error}")
                        empty_pose = [[0.0, 0.0, 0.0] for _ in range(17)]
                        pose_sequence.append(empty_pose)
                
                video_capture.release()
                
                if len(pose_sequence) == self.sequence_length:
                    label = row['label']
                    
                    outcome = 1
                    if 'outcome' in row:
                        outcome = 1 if row['outcome'].lower() in ['hit', 'success', 'landed'] else 0
                    
                    sequences.append(pose_sequence)
                    labels.append(label)
                    outcomes.append(outcome)
                    
                    valid_count += 1
                    if valid_count % 10 == 0:
                        print(f"Processed {valid_count} valid sequences")
                
            except Exception as e:
                print(f"Error processing {clip_path}: {str(e)}")
        
        if valid_count == 0:
            print("Error: No valid sequences were processed. Check your data and file paths.")
            return None
        
        print(f"Successfully processed {valid_count} clips.")
        
        metadata_path = os.path.join(self.output_directory, "metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump({
                'sequences': sequences,
                'labels': labels,
                'outcomes': outcomes,
                'classes': sorted(list(set(labels)))
            }, f, cls=NumpyEncoder)
        
        print(f"Metadata saved to {metadata_path}")
        return metadata_path
    
    def _extract_pose_sequence(self, video_path, start_frame, end_frame):
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                print(f"Warning: Could not open video {video_path}, skipping...")
                return None
            
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= start_frame:
                print(f"Warning: Video {video_path} has only {total_frames} frames, but start_frame is {start_frame}, skipping them")
                video_capture.release()
                return None
                
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            keypoints_sequence = []
            max_frames = min(end_frame - start_frame, 1000) #limit max frames to process
            
            while frame_count < max_frames:
                return_value, frame = video_capture.read()
                if not return_value:
                    break
                    
                if frame_count % 2 == 0: #process every second frame for efficiency
                    try:
                        _, people, _ = self.pose_detector.detect_pose(frame)
                        
                        person = people[0] if people else None
                        
                        if person and person.landmarks is not None:
                            keypoints_sequence.append(person.landmarks)
                    except Exception as e:
                        print(f"Warning: Error in pose detection for {video_path}, frame {start_frame + frame_count}: {e}")
                
                frame_count += 1
            
            video_capture.release()
            
            if len(keypoints_sequence) < 5:
                print(f"Warning: Not enough valid frames in {video_path} (only {len(keypoints_sequence)} found), skipping them")
                return None
            
            resampled = self._resample_sequence(keypoints_sequence)
            
            return np.array(resampled)
            
        except Exception as e:
            print(f"Warning: Exception processing {video_path}: {e}, skipping...")
            return None
    
    def _resample_sequence(self, sequence):
        if len(sequence) == self.sequence_length:
            return sequence
        
        if len(sequence) < self.sequence_length:
            while len(sequence) < self.sequence_length: #pad with last frame
                sequence.append(sequence[-1])
            return sequence
        
        indices = np.linspace(0, len(sequence) - 1, self.sequence_length)
        resampled = []
        
        for index in indices:
            if index.is_integer():
                resampled.append(sequence[int(index)])
            else:
                index_floor = int(np.floor(index))
                index_ceil = int(np.ceil(index))
                weight_ceil = index - index_floor
                
                interpolated = sequence[index_floor] * (1 - weight_ceil) + sequence[index_ceil] * weight_ceil #linear interpolation
                resampled.append(interpolated)
        return resampled


class StrikeDataset(Dataset):
    def __init__(self, metadata_path, sequence_length=15, transform=None, train=True, test_split=0.2):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.labels = data['labels']
        self.outcomes = data['outcomes']
        self.classes = data['classes']
        
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        
        total_samples = len(self.sequences)
        indices = list(range(total_samples))
        
        np.random.seed(30) #to create consisent train/test split
        np.random.shuffle(indices)
        
        split_index = int(np.floor(test_split * total_samples))
        
        if train:
            self.indices = indices[split_index:]
        else:
            self.indices = indices[:split_index]
            
        self.transform = transform
        self.train = train
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        true_index = self.indices[index]
        
        sequence = np.array(self.sequences[true_index])
        
        label = self.labels[true_index]
        label_index = self.class_to_idx[label]
        
        outcome_string = self.outcomes[true_index]
        outcome = 1 if outcome_string == 'successful' else 0
        
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.train:
            sequence = self._augment_sequence(sequence)
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        return sequence_tensor, label_index, outcome
    
    def _augment_sequence(self, sequence):
        noise = np.random.normal(0, 0.02, sequence.shape) #add subtle noise
        sequence = sequence + noise
        
        if random.random() < 0.4:
            scale_factor = random.uniform(0.7, 1.3)
            indices = np.linspace(0, len(sequence) - 1, int(len(sequence) * scale_factor))
            indices = np.clip(indices, 0, len(sequence) - 1)
            
            augmented_sequence = []
            for index in indices[:self.sequence_length]:
                augmented_sequence.append(sequence[int(index)])
            
            if len(augmented_sequence) < self.sequence_length:
                augmented_sequence.extend([sequence[-1]] * (self.sequence_length - len(augmented_sequence)))
            
            sequence = np.array(augmented_sequence[:self.sequence_length])
        
        if random.random() < 0.4:
            angle = random.uniform(-25, 25) * np.pi / 180
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            for i in range(len(sequence)):
                sequence[i, :, :2] = np.dot(sequence[i, :, :2], rotation_matrix.T) #rotate pose
        
        if random.random() < 0.4:
            scale_factor = random.uniform(0.7, 1.3)
            sequence[:, :, :2] *= scale_factor #scale keypoints
            
        if random.random() < 0.3:
            mask_joints = np.random.choice(sequence.shape[1], 
                                        size=int(sequence.shape[1] * 0.1), 
                                        replace=False)
            for joint in mask_joints:
                mask_frames = np.random.choice(sequence.shape[0], 
                                            size=int(sequence.shape[0] * 0.2), 
                                            replace=False)
                for frame in mask_frames:
                    sequence[frame, joint, 2] *= 0.5 #reduce confidence of random keypoints
        
        return sequence


def get_dataloaders(metadata_path, batch_size=32, sequence_length=15, num_workers=4, pin_memory=True):
    train_dataset = StrikeDataset(
        metadata_path=metadata_path,
        sequence_length=sequence_length,
        train=True
    )
    
    validation_dataset = StrikeDataset(
        metadata_path=metadata_path,
        sequence_length=sequence_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, validation_loader, train_dataset.classes