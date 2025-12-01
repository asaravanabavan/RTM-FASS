import os
import cv2
import numpy as np
import torch
import torch.nn as nn

def compute_relative_features(keypoints):
    sequence_length, number_of_people, number_of_joints, dimensions = keypoints.shape
    
    features = []
    
    for person_index in range(number_of_people):
        person_keypoints = keypoints[:, person_index, :, :]
        
        left_shoulder = person_keypoints[:, 5, :2]
        right_shoulder = person_keypoints[:, 6, :2]
        
        left_elbow = person_keypoints[:, 7, :2]
        right_elbow = person_keypoints[:, 8, :2]
        
        left_wrist = person_keypoints[:, 9, :2]
        right_wrist = person_keypoints[:, 10, :2]
        
        joint_velocities = np.zeros((sequence_length-1, number_of_joints, 2))
        for i in range(sequence_length-1):
            joint_velocities[i] = person_keypoints[i+1, :, :2] - person_keypoints[i, :, :2] #calculate velocity
        
        joint_acceleration = np.zeros((sequence_length-2, number_of_joints, 2))
        for i in range(sequence_length-2):
            joint_acceleration[i] = joint_velocities[i+1] - joint_velocities[i] #calculate acceleration
        
        person_features = {
            'wrist_velocity': joint_velocities[:-1, [9, 10], :],
            'elbow_velocity': joint_velocities[:-1, [7, 8], :],
            'ankle_velocity': joint_velocities[:-1, [15, 16], :],
            'knee_velocity': joint_velocities[:-1, [13, 14], :],
            'wrist_acceleration': joint_acceleration[:, [9, 10], :],
            'elbow_acceleration': joint_acceleration[:, [7, 8], :],
            'ankle_acceleration': joint_acceleration[:, [15, 16], :],
            'knee_acceleration': joint_acceleration[:, [13, 14], :]
        }
        
        features.append(person_features)
    
    return features

def detect_target_area(striker_keypoints, target_keypoints): #decides the body areas
    left_wrist = striker_keypoints[9, :2]
    right_wrist = striker_keypoints[10, :2]
    
    left_ankle = striker_keypoints[15, :2]
    right_ankle = striker_keypoints[16, :2]
    
    target_head = target_keypoints[0, :2]
    target_body = (target_keypoints[5, :2] + target_keypoints[6, :2]) / 2 #center of shoulders
    target_legs = (target_keypoints[11, :2] + target_keypoints[12, :2]) / 2 #center of hips
    
    wrist_to_head = min(np.linalg.norm(left_wrist - target_head), 
                        np.linalg.norm(right_wrist - target_head))
    wrist_to_body = min(np.linalg.norm(left_wrist - target_body),
                       np.linalg.norm(right_wrist - target_body))
    ankle_to_body = min(np.linalg.norm(left_ankle - target_body),
                        np.linalg.norm(right_ankle - target_body))
    ankle_to_legs = min(np.linalg.norm(left_ankle - target_legs),
                        np.linalg.norm(right_ankle - target_legs))
    
    if wrist_to_head < wrist_to_body:
        return 'head'
    elif min(wrist_to_body, ankle_to_body) < ankle_to_legs:
        return 'body'
    else:
        return 'legs'

try:
    from src.strike_model import StrikeNet
except ImportError: #sorts out that importerror from earlier
    print("Warning: Could not import StrikeNet class, model-based detection will be disabled")

class StrikeDetector:
    def __init__(self, model_path=None):
        self.strike_types = [
            'jab', 'cross', 'hook', 'uppercut', 
            'teep', 'roundhouse', 'knee', 'elbow'    
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.classes = checkpoint['classes']
                
                self.model = StrikeNet(
                    num_classes=len(self.classes),
                    sequence_length=15,
                    use_attention=True
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval() #set to evaluation mode
                print(f"Loaded StrikeNet from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.classes = None
        else:
            print("No StrikeNet model found, using heuristic detection")
            self.model = None
            self.classes = None
        
        self.frame_buffer = []
        self.buffer_size = 15 #sequence length for detection
        self.cooldown_frames = {0: 0, 1: 0} #prevent duplicate detections   
        self.cooldown_period = 10
        self.last_detections = {0: None, 1: None}
    
    def reset(self):
        self.frame_buffer = []
        self.cooldown_frames = {0: 0, 1: 0}
        self.last_detections = {0: None, 1: None}
    
    def process_frame(self, frame_index, people_landmarks):
        self.frame_buffer.append(people_landmarks)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        strikes = {0: [], 1: []}
        
        if len(self.frame_buffer) < self.buffer_size:
            return strikes
        
        for fighter_id in self.cooldown_frames:
            if self.cooldown_frames[fighter_id] > 0:
                self.cooldown_frames[fighter_id] -= 1
        
        if self.model is not None: #use neural network if available
            for fighter_id in [0, 1]:
                if self.cooldown_frames[fighter_id] > 0:
                    continue
                
                sequence = np.zeros((self.buffer_size, 17, 3))
                
                for i, people in enumerate(self.frame_buffer):
                    for person in people:
                        if person is not None and hasattr(person, 'id') and person.id == fighter_id:
                            if person.landmarks is not None:
                                sequence[i] = person.landmarks
                
                if np.sum(sequence[:, 0, 2]) < 5: #check if enough keypoints are visible
                    continue
                
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                sequence_tensor = sequence_tensor.to(self.device)
                
                with torch.no_grad():
                    class_predictions, outcome_predictions = self.model(sequence_tensor)
                    
                    class_probabilities = torch.softmax(class_predictions, dim=1)[0]
                    class_index = class_probabilities.argmax().item()
                    class_name = self.classes[class_index]
                    
                    outcome_probabilities = torch.softmax(outcome_predictions, dim=1)[0]
                    outcome_index = outcome_probabilities.argmax().item()
                    outcome = "successful" if outcome_index == 1 else "unsuccessful"
                    
                    confidence = class_probabilities[class_index].item()
                    
                    if confidence > 0.65: #threshold for detection
                        strikes[fighter_id].append({
                            'type': class_name,
                            'outcome': outcome,
                            'confidence': confidence,
                            'frame': frame_index
                        })
                        
                        self.cooldown_frames[fighter_id] = self.cooldown_period
        
        else: #fallback to heuristic detection if no model
            sequence = np.zeros((self.buffer_size, 2, 17, 3))
            
            for i, frame_people in enumerate(self.frame_buffer):
                for person in frame_people:
                    if person is not None and hasattr(person, 'id') and person.id in [0, 1]:
                        sequence[i, person.id] = person.landmarks
            
            features = compute_relative_features(sequence)
            
            for fighter_id in [0, 1]:
                if self.cooldown_frames[fighter_id] > 0:
                    continue
                
                if fighter_id < len(features):
                    fighter_features = features[fighter_id]
                    
                    wrist_velocity = fighter_features['wrist_velocity']
                    wrist_acceleration = fighter_features['wrist_acceleration']
                    
                    ankle_velocity = fighter_features['ankle_velocity']
                    ankle_acceleration = fighter_features['ankle_acceleration']
                    
                    left_wrist_velocity = np.linalg.norm(wrist_velocity[:, 0, :], axis=1)
                    right_wrist_velocity = np.linalg.norm(wrist_velocity[:, 1, :], axis=1)
                    
                    if np.max(left_wrist_velocity) > 20 and fighter_id == 0: #detect jab
                        strikes[fighter_id].append({
                            'type': 'jab',
                            'outcome': 'successful',
                            'confidence': min(1.0, np.max(left_wrist_velocity) / 30),
                            'frame': frame_index
                        })
                        self.cooldown_frames[fighter_id] = self.cooldown_period
                    
                    elif np.max(right_wrist_velocity) > 20 and fighter_id == 0: #detect cross
                        strikes[fighter_id].append({
                            'type': 'cross',
                            'outcome': 'successful',
                            'confidence': min(1.0, np.max(right_wrist_velocity) / 30),
                            'frame': frame_index
                        })
                        self.cooldown_frames[fighter_id] = self.cooldown_period
                    
                    left_ankle_velocity = np.linalg.norm(ankle_velocity[:, 0, :], axis=1)
                    right_ankle_velocity = np.linalg.norm(ankle_velocity[:, 1, :], axis=1)
                    
                    if np.max(left_ankle_velocity) > 25 and fighter_id == 0: #detect left kick
                        strikes[fighter_id].append({
                            'type': 'roundhouse',
                            'outcome': 'successful',
                            'confidence': min(1.0, np.max(left_ankle_velocity) / 35),
                            'frame': frame_index
                        })
                        self.cooldown_frames[fighter_id] = self.cooldown_period
                    
                    elif np.max(right_ankle_velocity) > 25 and fighter_id == 0: #detect right kick
                        strikes[fighter_id].append({
                            'type': 'roundhouse',
                            'outcome': 'successful',
                            'confidence': min(1.0, np.max(right_ankle_velocity) / 35),
                            'frame': frame_index
                        })
                        self.cooldown_frames[fighter_id] = self.cooldown_period
        
        return strikes