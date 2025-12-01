import numpy as np 

if not hasattr(np, 'dtypes'): #fix the dtype error since numpy latest broke
    class DtypesNamespace:
        def __init__(self):
            self.integer = np.integer
            self.floating = np.floating
            self.bool_ = np.bool_
            self.number = np.number
            self.generic = np.generic
                
    np.dtypes = DtypesNamespace()

import cv2
import torch
import time
import gc
from ultralytics import YOLO
from pathlib import Path

class Person:
    def __init__(self, id, bounding_box, landmarks, score):
        self.id = id
        self.bbox = bounding_box
        self.landmarks = landmarks
        self.score = score
        
        if bounding_box is not None:
            self.center = ((bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2)
        else:
            self.center = None

class PoseDetector:
    def __init__(self, 
                 model="yolov8x-pose.pt",
                 device=None,
                 conf_thresh=0.25,
                 kpt_thr=0.2):
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.conf_thresh = conf_thresh
        self.kpt_thr = kpt_thr
        
        print(f"Initializing PoseDetector on device: {self.device}")
        print(f"Using model: {model}, conf_thresh: {conf_thresh}, kpt_thr: {kpt_thr}")
        
        print(f"Loading YOLOv8 pose model: {model}")
        self.model = YOLO(model)
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA is available: {torch.cuda.is_available()}")
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        
        self.skeleton = [ #use for keypoint connections
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        self.pose_sequence = []
        self.max_sequence_length = 30
        
        self.prev_fighters = {} #stores track id to fighter id mapping
        self.fighter_centers = {} #stores fighter positions
        
        self.fighter_memory = {0: None, 1: None} #remembers fighters for occlusion handling
        self.lost_frames = {0: 0, 1: 0} #counts frames since fighter was last detected
        self.max_lost_frames = 30
        
        self.tracking_errors = 0
        self.max_tracking_errors = 5
        
        print("PoseDetector initialization complete")
    
    def batch_detect_pose(self, frames, batch_size=512):
        start_time = time.time()
        results = []
        
        process_batch_size = min(batch_size, 512)
        min_batch_size = 4
        
        def get_memory_usage(): #use this to check memory usage
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated() / 1e9,
                    'reserved': torch.cuda.memory_reserved() / 1e9,
                    'total': torch.cuda.get_device_properties(0).total_memory / 1e9
                }
            return {'allocated': 0, 'reserved': 0, 'total': 0}
        
        index = 0
        frames_processed = 0
        while index < len(frames):
            sub_batch = frames[index:index+process_batch_size]
            sub_batch_size = len(sub_batch)
            
            try:
                memory = get_memory_usage()
                print(f"Memory: {memory['allocated']:.2f}GB used, {memory['total']-memory['allocated']:.2f}GB free, batch: {sub_batch_size}")
                
                #resize large frames to improve processing speed
                preprocessed_batch = []
                for frame in sub_batch:
                    if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                        frame = cv2.resize(frame, (1280, 720))
                    preprocessed_batch.append(frame)
                
                model_results = self.model(preprocessed_batch, classes=0, conf=self.conf_thresh, verbose=False, stream=True)
                
                batch_results = []
                for frame_index, result in enumerate(model_results):
                    people = []
                    detected_fighter_ids = set()
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.cpu().numpy()
                        
                        keypoints = []
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints = result.keypoints.cpu().numpy()
                        
                        detections = []
                        for k, box in enumerate(boxes):
                            if hasattr(box, 'cls') and box.cls[0] == 0:
                                if hasattr(box, 'xyxy'):
                                    x1, y1, x2, y2 = box.xyxy[0]
                                else:
                                    x1, y1, x2, y2 = box.xywh[0]
                                    x2 = x1 + x2
                                    y2 = y1 + y2
                                
                                confidence = box.conf[0]
                                
                                keypoints_data = None
                                if k < len(keypoints):
                                    keypoints_data = keypoints[k].data[0]
                                
                                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'center': center,
                                    'score': confidence,
                                    'keypoints': keypoints_data
                                })
                        
                        #assign fighter ids (0 or 1) to detected people
                        for detection in detections:
                            fighter_id = None
                            
                            if 0 not in detected_fighter_ids:
                                fighter_id = 0
                            elif 1 not in detected_fighter_ids:
                                fighter_id = 1
                            else:
                                continue
                            
                            person = Person(
                                id=fighter_id,
                                bounding_box=detection['bbox'],
                                landmarks=detection['keypoints'],
                                score=detection['score']
                            )
                            people.append(person)
                            detected_fighter_ids.add(fighter_id)
                    
                    if frame_index < len(sub_batch):
                        frame = sub_batch[frame_index]
                        visualization_frame = self._visualize(frame, people)
                        batch_results.append((visualization_frame, people, 0.0))
                    
                    del people
                
                results.extend(batch_results)
                frames_processed += sub_batch_size
                
                #clean up memory
                del preprocessed_batch
                del batch_results
                
                index += sub_batch_size
                
                if frames_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    frames_per_second = frames_processed / elapsed if elapsed > 0 else 0
                    print(f"Processed {frames_processed}/{len(frames)} frames, {frames_per_second:.2f} FPS")
                
            except RuntimeError as e:  #handle cuda oom errors
                if "CUDA out of memory" in str(e):
                    process_batch_size = max(min_batch_size, process_batch_size // 2)
                    print(f"CUDA OOM error - reducing batch size to {process_batch_size}")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                else:
                    print(f"Error in batch detection: {e}")
                    for j in range(len(sub_batch)):
                        frame = sub_batch[j]
                        dummy = Person(
                            id=0,
                            bounding_box=[100, 100, 300, 400],
                            landmarks=np.zeros((17, 3)),
                            score=0.5
                        )
                        visualization_frame = self._visualize(frame, [dummy])
                        results.append((visualization_frame, [dummy], 0.0))
                    
                    index += sub_batch_size
                    frames_processed += sub_batch_size
            
            torch.cuda.empty_cache()
        
        end_time = time.time()
        total_time = end_time - start_time
        batch_fps = len(frames) / total_time if total_time > 0 else 0
        print(f"Batch processing completed: {len(frames)} frames in {total_time:.2f}s ({batch_fps:.2f} FPS)")
        
        return results, batch_fps
    
    def detect_pose(self, frame, use_tracking=True): #process single frame
        start_time = time.time()
        
        original_shape = frame.shape
        frame_processed = frame.copy()
        
        #resize large frames for faster processing
        if frame_processed.shape[0] > 1080 or frame_processed.shape[1] > 1920:
            frame_processed = cv2.resize(frame_processed, (1280, 720))
        
        try:
            if use_tracking and self.tracking_errors < self.max_tracking_errors:
                results = self.model.track(frame_processed, persist=True, classes=0, 
                                          conf=self.conf_thresh, verbose=False)
            else:
                results = self.model(frame_processed, classes=0, conf=self.conf_thresh, verbose=False)
                
            self.tracking_errors = 0
            
        except Exception as e:
            print(f"Error during detection/tracking: {e}")
            self.tracking_errors += 1
            
            try:
                results = self.model(frame_processed, classes=0, conf=self.conf_thresh, verbose=False)
            except Exception as e2:
                print(f"Error in fallback detection: {e2}")
                return frame, [], 0.0
        
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints = results[0].keypoints.cpu().numpy()
        
        people = []
        detected_fighter_ids = set()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.cpu().numpy()
            
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                keypoints = results[0].keypoints.cpu().numpy()
            else:
                keypoints = []
            
            detections = []
            for i, box in enumerate(boxes):
                if hasattr(box, 'cls') and box.cls[0] == 0:
                    if hasattr(box, 'xyxy'):
                        x1, y1, x2, y2 = box.xyxy[0]
                    else:
                        x1, y1, x2, y2 = box.xywh[0]
                        x2 = x1 + x2
                        y2 = y1 + y2
                    
                    confidence = box.conf[0]
                    
                    track_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    
                    keypoints_data = None
                    if i < len(keypoints):
                        keypoints_data = keypoints[i].data[0]
                    
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': center,
                        'score': confidence,
                        'track_id': track_id,
                        'keypoints': keypoints_data
                    })
            
            #assign consistent fighter ids based on tracking
            for detection in detections:
                fighter_id = None
                
                #if tracked person was seen before, use same id
                if detection['track_id'] is not None and detection['track_id'] in self.prev_fighters:
                    fighter_id = self.prev_fighters[detection['track_id']]
                
                if fighter_id is None:
                    if 0 not in detected_fighter_ids:
                        fighter_id = 0
                    elif 1 not in detected_fighter_ids:
                        fighter_id = 1
                    else:
                        continue
                
                person = Person(
                    id=fighter_id,
                    bounding_box=detection['bbox'],
                    landmarks=detection['keypoints'],
                    score=detection['score']
                )
                people.append(person)
                detected_fighter_ids.add(fighter_id)
                
                if detection['track_id'] is not None:
                    self.prev_fighters[detection['track_id']] = fighter_id
                    self.fighter_centers[detection['track_id']] = detection['center']
                
                self.fighter_memory[fighter_id] = person
                self.lost_frames[fighter_id] = 0
        
        #handle occlusion by using memory of previously seen fighters
        for fighter_id in [0, 1]:
            if fighter_id not in detected_fighter_ids and self.fighter_memory[fighter_id] is not None:
                self.lost_frames[fighter_id] += 1
                
                if self.lost_frames[fighter_id] <= self.max_lost_frames:
                    memory = self.fighter_memory[fighter_id]
                    person = Person(
                        id=fighter_id,
                        bounding_box=memory.bbox,
                        landmarks=memory.landmarks.copy() if memory.landmarks is not None else None,
                        score=memory.score * 0.9 #reduce confidence for each lost frame
                    )
                    people.append(person)
        
        self._update_pose_sequence(people)
        
        visualization_frame = self._visualize(frame, people)
        
        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        return visualization_frame, people, fps
    
    def detect_pose_without_tracking(self, frame):
        return self.detect_pose(frame, use_tracking=False)
    
    def _update_pose_sequence(self, people):
        #add new frame to sequence history
        self.pose_sequence.append(people)
        if len(self.pose_sequence) > self.max_sequence_length:
            self.pose_sequence.pop(0)
    
    def _visualize(self, frame, people):
        visualization_frame = frame.copy()
        
        colors = {
            0: (0, 255, 0),  #green for fighter 0
            1: (0, 0, 255)   #red for fighter 1
        }
        
        for person_idx, person in enumerate(people):
            fighter_id = person.id
            color = colors.get(fighter_id, (255, 255, 255))
            
            is_predicted = self.lost_frames.get(fighter_id, 0) > 0
            
            #draw bounding box
            if person.bbox is not None:
                x1, y1, x2, y2 = map(int, person.bbox)
                cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                
                if is_predicted:
                    cv2.putText(visualization_frame, f"Person {fighter_id} (predicted)", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    cv2.putText(visualization_frame, f"Person {fighter_id}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            #draw pose skeleton
            if person.landmarks is not None:
                keypoints = person.landmarks    #filter keypoints by confidence threshold
                valid_keypoints_mask = keypoints[:, 2] > self.kpt_thr
                valid_keypoints = keypoints[valid_keypoints_mask]
                
                if len(valid_keypoints) > 0:
                    points = valid_keypoints[:, :2].astype(np.int32)
                    for point in points:
                        cv2.circle(visualization_frame, (int(point[0]), int(point[1])), 5, color, -1)
                
                #connect joints with lines to form skeleton
                for i, (idx1, idx2) in enumerate(self.skeleton):
                    idx1 -= 1
                    idx2 -= 1
                    
                    if 0 <= idx1 < len(keypoints) and 0 <= idx2 < len(keypoints):
                        pt1 = keypoints[idx1]
                        pt2 = keypoints[idx2]
                        
                        if pt1[2] > self.kpt_thr and pt2[2] > self.kpt_thr:
                            cv2.line(visualization_frame, 
                                   (int(pt1[0]), int(pt1[1])), 
                                   (int(pt2[0]), int(pt2[1])),
                                   color, 2)
        
        return visualization_frame
    
    def get_pose_sequence(self, n_frames=30):
        sequence_length = min(n_frames, len(self.pose_sequence))
        sequence = np.zeros((sequence_length, 2, 17, 3))
        
        for i, people in enumerate(self.pose_sequence[-sequence_length:]):
            for person in people:
                fighter_id = person.id
                if fighter_id in [0, 1] and person.landmarks is not None:
                    sequence[i, fighter_id] = person.landmarks
        
        return sequence

    def safe_detect_pose(self, frame):
        try: #need for the numpy error
            return self.detect_pose(frame)
        except (AttributeError, ImportError, RuntimeError, cv2.error) as e:
            error_str = str(e).lower()
            if "numpy" in error_str or "dtypes" in error_str:
                print(f"Handling NumPy error: {e}")
                return self.detect_pose_without_tracking(frame)
            elif "lkpyramid" in error_str or "prevpyr" in error_str or "optical flow" in error_str:
                print(f"Handling optical flow error: {e}")
                self.tracking_errors += 1
                return self.detect_pose_without_tracking(frame)
            else:
                print(f"Handling unknown error: {e}")
                try:
                    return self.detect_pose_without_tracking(frame)
                except Exception as e2:
                    print(f"Error in fallback detection: {e2}")
                    class DummyPerson:
                        def __init__(self):
                            self.id = 0
                            self.bbox = [100, 100, 300, 400]
                            self.landmarks = np.zeros((17, 3))
                            self.score = 0.5
                            self.center = (200, 250)
                    
                    processed_frame = frame.copy()
                    cv2.rectangle(processed_frame, (100, 100), (300, 400), (0, 255, 0), 2)
                    cv2.putText(processed_frame, "Error Fallback", (100, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    return processed_frame, [DummyPerson()], 0.0