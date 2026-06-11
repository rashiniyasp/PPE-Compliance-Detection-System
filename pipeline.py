import numpy as np
from ultralytics import YOLO
import sys
import os
import urllib.request

# Import tracker modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tracking people'))
from tracker import SkeletonTracker
from config import Config
from skeleton_features import SkeletonFeatureExtractor

class PPEPipeline:
    def __init__(self, tracking_method='skeleton_yolo', detection_model_path='from_lab/runs/detect/runs/train/ppe_detection/weights/best.pt', pose_model_path='yolo11x-pose.pt', conf=0.3, fps=30.0, frame_width=1920, frame_height=1080):
        self.tracking_method = tracking_method
        self.conf = conf
        self.fps = fps
        self.frame_idx = 0
        self.frame_width = frame_width
        self.frame_height = frame_height

        print(f"Loading PPE Detection Model... (Method: {self.tracking_method})")
        self.det_model = YOLO(detection_model_path)
        
        self.class_names = self.det_model.names
        self.ppe_target_map = {
            'helmet': 'head',
            'goggles': 'head',
            'vest': 'torso',
            'gloves': 'wrists',
            'boots': 'ankles'
        }

        # Method specific initializations
        if self.tracking_method == 'skeleton_yolo':
            print("Loading YOLO-Pose Model...")
            self.pose_model = YOLO(pose_model_path)
            tracker_config = Config()
            self.tracker = SkeletonTracker(tracker_config, ablation_mode=3, fps=self.fps, frame_width=frame_width, frame_height=frame_height)
            self.feature_extractor = SkeletonFeatureExtractor(conf_threshold=0.5)

        elif self.tracking_method == 'skeleton_mediapipe':
            print("Loading MediaPipe Pose Model...")
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if not exists
            mp_model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(mp_model_path):
                print("Downloading MediaPipe model...")
                urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', mp_model_path)

            base_options = python.BaseOptions(model_asset_path=mp_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=20,
                min_pose_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                output_segmentation_masks=False)
            self.mp_detector = vision.PoseLandmarker.create_from_options(options)
            self.mp_Image = mp.Image
            self.mp_ImageFormat = mp.ImageFormat

            tracker_config = Config()
            self.tracker = SkeletonTracker(tracker_config, ablation_mode=3, fps=self.fps, frame_width=frame_width, frame_height=frame_height)
            self.feature_extractor = SkeletonFeatureExtractor(conf_threshold=0.5)
            
            # MP to COCO Map
            self.mp_to_coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        elif self.tracking_method == 'traditional_iou':
            print("Using Traditional BoT-SORT IoU Tracking...")
            # We don't need a separate pose model or skeleton tracker for this.
            self.track_history = {} # track_id -> person_data

    def process_frame(self, frame):
        if self.tracking_method == 'skeleton_yolo':
            return self._process_skeleton_yolo(frame)
        elif self.tracking_method == 'skeleton_mediapipe':
            return self._process_skeleton_mediapipe(frame)
        elif self.tracking_method == 'traditional_iou':
            return self._process_traditional_iou(frame)

    def _process_skeleton_yolo(self, frame):
        pose_results = self.pose_model.predict(frame, conf=self.conf, verbose=False)
        det_results = self.det_model.predict(frame, conf=self.conf, verbose=False)
        
        persons = []
        ppes = []
        
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
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
                
            tracks = self.tracker.update(detections, self.frame_idx)
            self.frame_idx += 1
            
            for track in tracks:
                if track.missed == 0:
                    persons.append({
                        'id': track.track_id,
                        'bbox': track.bbox,
                        'keypoints': track.keypoints,
                        'assigned_ppe': [],
                        'ppe_details': []
                    })
        else:
            self.tracker.update([], self.frame_idx)
            self.frame_idx += 1
                
        if len(det_results) > 0 and det_results[0].boxes is not None:
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
            det_classes = det_results[0].boxes.cls.cpu().numpy()
            det_confs = det_results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(det_boxes, det_classes, det_confs):
                class_name = self.class_names[int(cls)]
                if class_name in ['Person', 'none', 'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']:
                    continue
                ppes.append({'bbox': box, 'class': class_name, 'conf': conf})
                
        return self.associate_ppe_keypoints(persons, ppes)

    def _process_skeleton_mediapipe(self, frame):
        import mediapipe as mp
        # Convert BGR (cv2) to RGB and make it contiguous for MediaPipe
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        mp_image = self.mp_Image(image_format=self.mp_ImageFormat.SRGB, data=rgb_frame)
        pose_result = self.mp_detector.detect(mp_image)
        
        det_results = self.det_model.predict(frame, conf=self.conf, verbose=False)
        
        persons = []
        ppes = []
        
        detections = []
        if pose_result.pose_landmarks:
            for pose_idx, landmarks in enumerate(pose_result.pose_landmarks):
                # Convert 33 MP landmarks to 17 COCO landmarks
                # Format: [17, 3] (x, y, conf)
                coco_kpts = np.zeros((17, 3))
                for i, mp_idx in enumerate(self.mp_to_coco):
                    lm = landmarks[mp_idx]
                    coco_kpts[i] = [lm.x * self.frame_width, lm.y * self.frame_height, lm.visibility]
                
                # Estimate bounding box from keypoints
                valid_kpts = coco_kpts[coco_kpts[:, 2] > 0.1]
                if len(valid_kpts) > 0:
                    x_min = np.min(valid_kpts[:, 0])
                    y_min = np.min(valid_kpts[:, 1])
                    x_max = np.max(valid_kpts[:, 0])
                    y_max = np.max(valid_kpts[:, 1])
                    
                    # Add margin
                    margin = 20
                    box = np.array([max(0, x_min - margin), max(0, y_min - margin), min(self.frame_width, x_max + margin), min(self.frame_height, y_max + margin)])
                    bbox_height = box[3] - box[1]
                    
                    # Use presence as a confidence stand-in
                    conf = np.mean(valid_kpts[:, 2])
                    
                    features, mask = self.feature_extractor.extract(coco_kpts, bbox_height)
                    detections.append({
                        'bbox': box,
                        'keypoints': coco_kpts,
                        'features': features,
                        'mask': mask,
                        'conf': conf
                    })
                    
        tracks = self.tracker.update(detections, self.frame_idx)
        self.frame_idx += 1
        
        for track in tracks:
            if track.missed == 0:
                persons.append({
                    'id': track.track_id,
                    'bbox': track.bbox,
                    'keypoints': track.keypoints,
                    'assigned_ppe': [],
                    'ppe_details': []
                })
                
        if len(det_results) > 0 and det_results[0].boxes is not None:
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
            det_classes = det_results[0].boxes.cls.cpu().numpy()
            det_confs = det_results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(det_boxes, det_classes, det_confs):
                class_name = self.class_names[int(cls)]
                if class_name in ['Person', 'none', 'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']:
                    continue
                ppes.append({'bbox': box, 'class': class_name, 'conf': conf})
                
        return self.associate_ppe_keypoints(persons, ppes)

    def _process_traditional_iou(self, frame):
        # Run detection and tracking native in YOLO
        results = self.det_model.track(frame, persist=True, tracker="botsort.yaml", conf=self.conf, verbose=False)
        
        persons = []
        ppes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Extract IDs if available
            ids = results[0].boxes.id
            if ids is not None:
                ids = ids.cpu().numpy()
            else:
                ids = [-1] * len(boxes)

            for box, cls, conf, track_id in zip(boxes, classes, confs, ids):
                class_name = self.class_names[int(cls)]
                if class_name == 'Person':
                    if track_id != -1:
                        persons.append({
                            'id': int(track_id),
                            'bbox': box,
                            'assigned_ppe': [],
                            'ppe_details': []
                        })
                elif class_name not in ['none', 'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']:
                    ppes.append({'bbox': box, 'class': class_name, 'conf': conf})
                    
        # Associate PPE using simple Bounding Box IoU
        for ppe in ppes:
            ppe_box = ppe['bbox']
            best_person_idx = -1
            max_iou = 0.0
            
            for p_idx, person in enumerate(persons):
                iou = self._calculate_iou(person['bbox'], ppe_box)
                # Alternatively, check if PPE is mostly INSIDE the person box
                # Intersection area / PPE area
                inter_area = self._calculate_intersection_area(person['bbox'], ppe_box)
                ppe_area = (ppe_box[2]-ppe_box[0]) * (ppe_box[3]-ppe_box[1])
                containment = inter_area / (ppe_area + 1e-6)
                
                if containment > 0.3 and containment > max_iou:
                    max_iou = containment
                    best_person_idx = p_idx
                    
            if best_person_idx != -1:
                persons[best_person_idx]['assigned_ppe'].append(ppe['class'])
                persons[best_person_idx]['ppe_details'].append({
                    'class': ppe['class'],
                    'conf': float(ppe['conf']),
                    'bbox': ppe['bbox']
                })
                
        return persons

    def _calculate_intersection_area(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        return interArea

    def _calculate_iou(self, boxA, boxB):
        interArea = self._calculate_intersection_area(boxA, boxB)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def associate_ppe_keypoints(self, persons, ppes):
        # Existing logic for keypoint association
        for ppe in ppes:
            ppe_box = ppe['bbox']
            best_person_idx = -1
            min_dist = float('inf')
            
            ppe_center = np.array([(ppe_box[0] + ppe_box[2]) / 2, (ppe_box[1] + ppe_box[3]) / 2])
            
            for p_idx, person in enumerate(persons):
                kp = person['keypoints']
                target_part = self.ppe_target_map.get(ppe['class'])
                
                relevant_points = []
                if target_part == 'head':
                    relevant_points = kp[0:5]
                elif target_part == 'torso':
                    relevant_points = kp[5:7]
                elif target_part == 'wrists':
                    relevant_points = kp[9:11]
                elif target_part == 'ankles':
                    relevant_points = kp[15:17]
                    
                valid_points = [p[:2] for p in relevant_points if p[2] > 0.5]
                
                if valid_points:
                    part_center = np.mean(valid_points, axis=0)
                    dist = np.linalg.norm(part_center - ppe_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_person_idx = p_idx
                        
            if best_person_idx != -1:
                if min_dist < 150:
                    persons[best_person_idx]['assigned_ppe'].append(ppe['class'])
                    persons[best_person_idx]['ppe_details'].append({
                        'class': ppe['class'],
                        'conf': float(ppe['conf']),
                        'bbox': ppe['bbox']
                    })
                    
        return persons

    def check_compliance(self, persons):
        required_ppe = {'helmet', 'vest', 'gloves', 'boots'}
        results = []
        
        for person in persons:
            detected = set(person['assigned_ppe'])
            missing = required_ppe - detected
            
            conf_strs = []
            for d in person.get('ppe_details', []):
                conf_strs.append(f"{d['class']}:{d['conf']:.2f}")
            
            results.append({
                'person_id': person['id'],
                'bbox': person['bbox'],
                'keypoints': person.get('keypoints', None),
                'detected_ppe': list(detected),
                'missing_ppe': list(missing),
                'ppe_details': person.get('ppe_details', []),
                'conf_str': ', '.join(conf_strs) if conf_strs else '—',
                'alert': len(missing) > 0
            })
            
        return results
