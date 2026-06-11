import numpy as np
from ultralytics import YOLO

class PPEPipeline:
    def __init__(self, detection_model_path, pose_model_path='yolo11n-pose.pt', conf=0.3):
        print("Loading PPE Detection Model...")
        self.det_model = YOLO(detection_model_path)
        print("Loading YOLO-Pose Model...")
        self.pose_model = YOLO(pose_model_path)
        self.conf = conf
        
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
            
            for i, (box, keypoints) in enumerate(zip(person_boxes, keypoints_data)):
                persons.append({
                    'id': i,
                    'bbox': box,
                    'keypoints': keypoints,
                    'assigned_ppe': []
                })
                
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
            
            results.append({
                'person_id': person['id'],
                'bbox': person['bbox'],
                'detected_ppe': list(detected),
                'missing_ppe': list(missing),
                'alert': len(missing) > 0
            })
            
        return results
