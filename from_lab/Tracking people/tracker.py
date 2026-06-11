import numpy as np
from scipy.optimize import linear_sum_assignment
from skeleton_features import compute_feature_distance

def compute_iou(box1, box2):
    """ Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]. """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

class Track:
    def __init__(self, track_id, bbox, keypoints, features, mask, fps=None, start_frame=0):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.keypoints = keypoints
        self.features = features
        self.mask = mask
        self.conf = 1.0 # Set appropriately if needed
        
        self.age = 1
        self.hits = 1
        self.missed = 0
        self.is_matched = True
        self.assigned_cost = 0.0
        
        # Individual cost components from the last match
        self.last_iou_cost = 0.0
        self.last_center_cost = 0.0
        self.last_feature_cost = 0.0
        
        self.fps = fps if fps is not None and fps > 0 else None
        self.center = self.get_center(bbox)
        
        # History for velocity (center, timestamp_in_sec)
        self.history = [] 
        time_sec = start_frame / self.fps if self.fps else start_frame
        self.history.append((self.center, time_sec))
        self.velocity = np.array([0.0, 0.0])

    def get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2])/2.0, (bbox[1] + bbox[3])/2.0])

    def update(self, bbox, keypoints, features, mask, conf, frame_idx, cost_dict, velocity_window):
        self.bbox = bbox
        self.keypoints = keypoints
        self.features = features
        self.mask = mask
        self.conf = conf
        self.hits += 1
        self.missed = 0
        self.age += 1
        self.is_matched = True
        self.assigned_cost = cost_dict.get('total', 0.0)
        self.last_iou_cost = cost_dict.get('iou', 0.0)
        self.last_center_cost = cost_dict.get('center', 0.0)
        self.last_feature_cost = cost_dict.get('feature', 0.0)
        
        self.center = self.get_center(bbox)
        time_sec = frame_idx / self.fps if self.fps else frame_idx
        self.history.append((self.center, time_sec))
        
        if len(self.history) > velocity_window:
            self.history.pop(0)
            
        if len(self.history) > 1:
            c1, t1 = self.history[0]
            c2, t2 = self.history[-1]
            dt = t2 - t1
            if dt > 0:
                self.velocity = (c2 - c1) / dt

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self.is_matched = False
        self.assigned_cost = 0.0
        self.last_iou_cost = 0.0
        self.last_center_cost = 0.0
        self.last_feature_cost = 0.0

class SkeletonTracker:
    def __init__(self, config, ablation_mode, fps, frame_width, frame_height):
        self.config = config
        self.ablation_mode = ablation_mode
        self.fps = fps
        self.frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        
        # Override max_age to be time-based if fps is available
        if self.fps and self.fps > 0:
            target_sec = getattr(self.config, 'max_age_sec', 4.0)
            self.max_age_frames = int(self.fps * target_sec)
        else:
            self.max_age_frames = getattr(self.config, 'max_age', 30)

        self.tracks = []
        self.next_id = 1
        self.iou_w, self.feat_w, self.center_w = config.get_weights(ablation_mode)

    def update(self, detections, frame_idx):
        """
        detections: list of dicts:
            {'bbox': [x1, y1, x2, y2], 'keypoints': array(17,3), 'features': array, 'mask': array, 'conf': float}
        """
        if len(self.tracks) == 0:
            for det in detections:
                new_track = Track(self.next_id, det['bbox'], det['keypoints'], det['features'], det['mask'], self.fps, frame_idx)
                new_track.conf = det['conf']
                self.tracks.append(new_track)
                self.next_id += 1
            return self.tracks

        if len(detections) == 0:
            for trk in self.tracks:
                trk.mark_missed()
        else:
            # Build cost matrix
            num_tracks = len(self.tracks)
            num_dets = len(detections)
            cost_matrix = np.zeros((num_tracks, num_dets))
            
            # Store individual costs for logging
            cost_components = [[{} for _ in range(num_dets)] for _ in range(num_tracks)]

            for t, trk in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    # 1. IoU cost
                    iou = compute_iou(trk.bbox, det['bbox'])
                    iou_cost = 1.0 - iou
                    
                    # 2. Center distance cost
                    det_center = np.array([(det['bbox'][0] + det['bbox'][2])/2, (det['bbox'][1] + det['bbox'][3])/2])
                    # If we have velocity, we could predict center, but baseline uses pure distance
                    dist = np.linalg.norm(trk.center - det_center)
                    center_cost = dist / self.frame_diagonal
                    
                    # 3. Feature distance cost
                    feat_cost = compute_feature_distance(trk.features, trk.mask, det['features'], det['mask'])
                    
                    total_cost = (self.iou_w * iou_cost) + (self.center_w * center_cost) + (self.feat_w * feat_cost)
                    cost_matrix[t, d] = total_cost
                    
                    cost_components[t][d] = {
                        'total': total_cost,
                        'iou': iou_cost,
                        'center': center_cost,
                        'feature': feat_cost
                    }

            # Solve assignment
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            unmatched_tracks = set(range(num_tracks))
            unmatched_dets = set(range(num_dets))
            
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] > self.config.max_cost:
                    continue # Reject assignment
                
                trk = self.tracks[r]
                det = detections[c]
                trk.update(
                    det['bbox'], det['keypoints'], det['features'], det['mask'], det['conf'], 
                    frame_idx, cost_components[r][c], self.config.velocity_window
                )
                
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
                
            for r in unmatched_tracks:
                self.tracks[r].mark_missed()
                
            for c in unmatched_dets:
                det = detections[c]
                new_track = Track(self.next_id, det['bbox'], det['keypoints'], det['features'], det['mask'], self.fps, frame_idx)
                new_track.conf = det['conf']
                self.tracks.append(new_track)
                self.next_id += 1

        # Delete old tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age_frames]
        
        return self.tracks
