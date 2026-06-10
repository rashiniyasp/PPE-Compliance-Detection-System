import numpy as np
from ultralytics import YOLO
import sys
import os

# Import tracker modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tracking people'))
from tracker import SkeletonTracker
from config import Config
from skeleton_features import SkeletonFeatureExtractor

class PPEPipeline:
    def __init__(self, detection_model_path=r'r:\SYSTECH\Project\from_lab\runs\detect\runs\train\ppe_detection\weights\best.pt', pose_model_path='yolo11x-pose.pt', conf=0.3, fps=30.0, frame_width=1920, frame_height=1080):
        print("Loading PPE Detection Model...")
        self.det_model = YOLO(detection_model_path)
        print("Loading YOLO-Pose Model...")
        self.pose_model = YOLO(pose_model_path)
        self.conf = conf
        
        self.fps = fps
        self.frame_idx = 0
        
        # Initialize Tracker & Feature Extractor
        tracker_config = Config()
        self.tracker = SkeletonTracker(tracker_config, ablation_mode=3, fps=self.fps, frame_width=frame_width, frame_height=frame_height)
        self.feature_extractor = SkeletonFeatureExtractor(conf_threshold=0.5)
        
        # Define PPE classes based on your data.yaml
        self.class_names = self.det_model.names
        # Ensure we have the basic expected mapping
        self.ppe_target_map = {
            'helmet': 'head',
            'goggles': 'head',
            'vest': 'torso',
            'gloves': 'wrists',
            'boots': 'ankles'
        }

    def process_frame(self, frame):
        """
        Runs both pose estimation and PPE detection on a single frame,
        then associates PPE to persons based on keypoints.
        """
        # 1. Detect Persons and Keypoints
        pose_results = self.pose_model.predict(frame, conf=self.conf, verbose=False)
        
        # 2. Detect PPE
        det_results = self.det_model.predict(frame, conf=self.conf, verbose=False)
        
        persons = []
        ppes = []
        
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            # Extract keypoints and person boxes
            keypoints_data = pose_results[0].keypoints.data.cpu().numpy() # [N, 17, 3] (x, y, conf)
            person_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            person_confs = pose_results[0].boxes.conf.cpu().numpy()
            
            detections = []
            for box, keypoints, conf in zip(person_boxes, keypoints_data, person_confs):
                bbox_height = box[3] - box[1]
                features, mask = self.feature_extractor.extract(keypoints, bbox_height)
                
                detections.append({
                    'bbox': box,
                    'keypoints': keypoints,
                    'features': features,
                    'mask': mask,
                    'conf': conf
                })
                
            # Update Tracker
            tracks = self.tracker.update(detections, self.frame_idx)
            self.frame_idx += 1
            
            for track in tracks:
                if track.missed == 0: # Only active tracks
                    persons.append({
                        'id': track.track_id,
                        'bbox': track.bbox,
                        'keypoints': track.keypoints,
                        'assigned_ppe': [],
                        'ppe_details': []  # List of {'class', 'conf', 'bbox'}
                    })
        else:
            # Update tracker with empty detections to increment age
            self.tracker.update([], self.frame_idx)
            self.frame_idx += 1
                
        if len(det_results) > 0 and det_results[0].boxes is not None:
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
            det_classes = det_results[0].boxes.cls.cpu().numpy()
            det_confs = det_results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(det_boxes, det_classes, det_confs):
                class_name = self.class_names[int(cls)]
                if class_name in ['Person', 'none', 'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']:
                    continue # Skip non-PPE detections or handle negative classes separately
                    
                ppes.append({
                    'bbox': box,
                    'class': class_name,
                    'conf': conf
                })
                
        # 3. Associate PPE to Persons
        associated_data = self.associate_ppe(persons, ppes)
        return associated_data

    def associate_ppe(self, persons, ppes):
        """
        Associates each piece of PPE to the person whose keypoints fall within 
        or are closest to the PPE bounding box.
        """
        for ppe in ppes:
            ppe_box = ppe['bbox']
            best_person_idx = -1
            min_dist = float('inf')
            
            # Simple association: Which person's relevant keypoint is closest to the center of the PPE box?
            ppe_center = np.array([(ppe_box[0] + ppe_box[2]) / 2, (ppe_box[1] + ppe_box[3]) / 2])
            
            for p_idx, person in enumerate(persons):
                kp = person['keypoints']
                target_part = self.ppe_target_map.get(ppe['class'])
                
                # Get relevant keypoints based on body part
                relevant_points = []
                if target_part == 'head':
                    relevant_points = kp[0:5] # Nose, Eyes, Ears
                elif target_part == 'torso':
                    relevant_points = kp[5:7] # Shoulders
                elif target_part == 'wrists':
                    relevant_points = kp[9:11] # Wrists
                elif target_part == 'ankles':
                    relevant_points = kp[15:17] # Ankles
                    
                # Calculate distance if keypoints are confident enough
                valid_points = [p[:2] for p in relevant_points if p[2] > 0.5]
                
                if valid_points:
                    # Average position of valid relevant keypoints
                    part_center = np.mean(valid_points, axis=0)
                    dist = np.linalg.norm(part_center - ppe_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_person_idx = p_idx
                        
            if best_person_idx != -1:
                # Add a threshold to avoid associating PPE to someone far away
                if min_dist < 150: # Adjust based on image scale
                    persons[best_person_idx]['assigned_ppe'].append(ppe['class'])
                    persons[best_person_idx]['ppe_details'].append({
                        'class': ppe['class'],
                        'conf': float(ppe['conf']),
                        'bbox': ppe['bbox']
                    })
                    
        return persons

    def check_compliance(self, persons):
        """
        Checks if the person has all required PPE.
        """
        required_ppe = {'helmet', 'vest', 'gloves', 'boots'}
        results = []
        
        for person in persons:
            detected = set(person['assigned_ppe'])
            missing = required_ppe - detected
            
            # Build confidence string like "helmet:0.92, vest:0.87"
            conf_strs = []
            for d in person.get('ppe_details', []):
                conf_strs.append(f"{d['class']}:{d['conf']:.2f}")
            
            results.append({
                'person_id': person['id'],
                'bbox': person['bbox'],
                'detected_ppe': list(detected),
                'missing_ppe': list(missing),
                'ppe_details': person.get('ppe_details', []),
                'conf_str': ', '.join(conf_strs) if conf_strs else '—',
                'alert': len(missing) > 0
            })
            
        return results
