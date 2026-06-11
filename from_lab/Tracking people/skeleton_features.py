import numpy as np
import math

class SkeletonFeatureExtractor:
    # COCO Keypoint Indices
    NOSE = 0
    L_EYE, R_EYE = 1, 2
    L_EAR, R_EAR = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

    def calculate_angle(self, p1, p2, p3):
        """ Calculate angle between three points (p2 is the vertex). Returns angle in degrees [0, 180]. """
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isnan(p3)):
            return np.nan
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def extract(self, keypoints, bbox_height):
        """
        Extract features from 17 keypoints.
        keypoints: shape (17, 3) where each row is [x, y, conf]
        bbox_height: fallback for torso scale
        
        Returns:
            features: np.array of feature values
            mask: np.array of 1s and 0s indicating valid features
        """
        feats = []
        mask = []

        # Helper to get point and its validity
        def pt(idx):
            x, y, c = keypoints[idx]
            if c > self.conf_threshold:
                return np.array([x, y]), 1.0
            return np.array([np.nan, np.nan]), 0.0

        p_l_sh, c_l_sh = pt(self.L_SHOULDER)
        p_r_sh, c_r_sh = pt(self.R_SHOULDER)
        p_l_hip, c_l_hip = pt(self.L_HIP)
        p_r_hip, c_r_hip = pt(self.R_HIP)
        
        p_l_el, c_l_el = pt(self.L_ELBOW)
        p_r_el, c_r_el = pt(self.R_ELBOW)
        p_l_wr, c_l_wr = pt(self.L_WRIST)
        p_r_wr, c_r_wr = pt(self.R_WRIST)
        
        p_l_kn, c_l_kn = pt(self.L_KNEE)
        p_r_kn, c_r_kn = pt(self.R_KNEE)
        p_l_an, c_l_an = pt(self.L_ANKLE)
        p_r_an, c_r_an = pt(self.R_ANKLE)

        # 1. Torso Scale & Origin
        mid_hip = None
        if c_l_hip and c_r_hip:
            mid_hip = (p_l_hip + p_r_hip) / 2.0
            
        mid_shoulder = None
        if c_l_sh and c_r_sh:
            mid_shoulder = (p_l_sh + p_r_sh) / 2.0
            
        torso_scale = bbox_height
        if mid_hip is not None and mid_shoulder is not None:
            dist = np.linalg.norm(mid_shoulder - mid_hip)
            if dist > bbox_height * 0.1: # Reasonable torso
                torso_scale = dist
        torso_scale = max(torso_scale, 1e-6)

        # Helper to add a feature safely
        def add_feat(val, valid):
            if valid and not np.isnan(val):
                feats.append(val)
                mask.append(1.0)
            else:
                feats.append(0.0)
                mask.append(0.0)

        # Helper to add distance ratio
        def add_dist_ratio(p1, c1, p2, c2):
            if c1 and c2:
                dist = np.linalg.norm(p1 - p2) / torso_scale
                add_feat(dist, True)
            else:
                add_feat(0.0, False)

        # 2. Bone Length Ratios (10 features)
        add_dist_ratio(p_l_sh, c_l_sh, p_l_el, c_l_el) # Upper arm L
        add_dist_ratio(p_r_sh, c_r_sh, p_r_el, c_r_el) # Upper arm R
        add_dist_ratio(p_l_el, c_l_el, p_l_wr, c_l_wr) # Lower arm L
        add_dist_ratio(p_r_el, c_r_el, p_r_wr, c_r_wr) # Lower arm R
        add_dist_ratio(p_l_hip, c_l_hip, p_l_kn, c_l_kn) # Upper leg L
        add_dist_ratio(p_r_hip, c_r_hip, p_r_kn, c_r_kn) # Upper leg R
        add_dist_ratio(p_l_kn, c_l_kn, p_l_an, c_l_an) # Lower leg L
        add_dist_ratio(p_r_kn, c_r_kn, p_r_an, c_r_an) # Lower leg R
        add_dist_ratio(p_l_sh, c_l_sh, p_r_sh, c_r_sh) # Shoulder width
        add_dist_ratio(p_l_hip, c_l_hip, p_r_hip, c_r_hip) # Hip width
        
        # Helper to add angle
        def add_ang(p1, c1, p2, c2, p3, c3):
            if c1 and c2 and c3:
                ang = self.calculate_angle(p1, p2, p3)
                add_feat(ang / 180.0, True) # Normalize to [0, 1]
            else:
                add_feat(0.0, False)

        # 3. Joint Angles (8 features)
        add_ang(p_l_sh, c_l_sh, p_l_el, c_l_el, p_l_wr, c_l_wr) # L elbow
        add_ang(p_r_sh, c_r_sh, p_r_el, c_r_el, p_r_wr, c_r_wr) # R elbow
        add_ang(p_l_hip, c_l_hip, p_l_kn, c_l_kn, p_l_an, c_l_an) # L knee
        add_ang(p_r_hip, c_r_hip, p_r_kn, c_r_kn, p_r_an, c_r_an) # R knee
        add_ang(p_l_el, c_l_el, p_l_sh, c_l_sh, p_l_hip, c_l_hip) # L shoulder
        add_ang(p_r_el, c_r_el, p_r_sh, c_r_sh, p_r_hip, c_r_hip) # R shoulder
        add_ang(p_l_kn, c_l_kn, p_l_hip, c_l_hip, p_l_sh, c_l_sh) # L hip
        add_ang(p_r_kn, c_r_kn, p_r_hip, c_r_hip, p_r_sh, c_r_sh) # R hip

        # 4. Normalized Coordinates (17 * 2 = 34 features)
        # Using mid_hip as origin if available, else bbox center (0,0 is fine as fallback if we don't have mid_hip, but usually we just skip them if mid_hip is missing)
        # Actually it's better to ignore normalized coords if mid_hip is missing.
        for i in range(17):
            p, c = pt(i)
            if c and mid_hip is not None:
                norm_p = (p - mid_hip) / torso_scale
                add_feat(norm_p[0], True)
                add_feat(norm_p[1], True)
            else:
                add_feat(0.0, False)
                add_feat(0.0, False)

        return np.array(feats), np.array(mask)

def compute_feature_distance(f1, m1, f2, m2):
    """ Calculate mask-aware feature distance (L1 distance). """
    mask = m1 * m2
    weight = np.sum(mask)
    if weight == 0:
        return 1.0 # Max distance if no overlapping valid features
    
    dist = np.sum(np.abs(f1 - f2) * mask) / weight
    # Optional: could clip or normalize, but L1 of our features (mostly ratios [0, ~2] and angles [0, 1] and coords) 
    # typically falls nicely between 0 and 1-2. Let's clip to [0, 1] just to bound the cost.
    return np.clip(dist, 0.0, 1.0)
