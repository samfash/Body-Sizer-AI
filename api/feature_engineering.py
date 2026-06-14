import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

LM = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28,
}


def variance_of_laplacian_gray(image):
    """Return variance of Laplacian (blurriness measure). Assumes BGR input."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def bbox_from_landmarks(kps, img_w, img_h):
    coords = np.array([[p[0], p[1]] for p in kps if p is not None])
    if coords.size == 0:
        return None

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)
    return x_min, y_min, x_max, y_max


def extract_landmarks_from_bgr_image(image_bgr):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = POSE.process(image_rgb)
    if not res.pose_landmarks:
        return None

    kps = []
    for lm in res.pose_landmarks.landmark:
        kps.append([lm.x * w, lm.y * h, lm.z * w, lm.visibility])
    return kps


def compute_features_from_landmarks(kps, image):
    img_h, img_w = image.shape[:2]

    if not kps:
        return None

    def dist(a, b):
        return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

    try:
        ls = kps[LM["left_shoulder"]]
        rs = kps[LM["right_shoulder"]]
        lh = kps[LM["left_hip"]]
        rh = kps[LM["right_hip"]]
        lw = kps[LM["left_wrist"]]
        rw = kps[LM["right_wrist"]]
        la = kps[LM["left_ankle"]]
        ra = kps[LM["right_ankle"]]
    except Exception:
        return None

    vis_list = [kp[3] for kp in [ls, rs, lh, rh, lw, rw, la, ra]]
    avg_visibility = float(np.mean(vis_list)) if len(vis_list) > 0 else 0.0

    shoulder_px = dist(ls, rs)
    hip_px = dist(lh, rh)
    sleeve_px = (dist(ls, lw) + dist(rs, rw)) / 2.0
    mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    torso_px = dist((mid_shoulder[0], mid_shoulder[1]), (mid_hip[0], mid_hip[1]))

    lowest_ank_y = max(la[1], ra[1])
    height_px = abs(lowest_ank_y - mid_shoulder[1]) + 1e-6

    shoulder_z = float((ls[2] + rs[2]) / 2.0)
    hip_z = float((lh[2] + rh[2]) / 2.0)

    bbox = bbox_from_landmarks(kps, img_w, img_h)
    if bbox is None:
        bbox_area = 0.0
    else:
        x1, y1, x2, y2 = bbox
        bbox_area = max(0.0, (x2 - x1) * (y2 - y1))

    img_area = float(img_w * img_h)
    bbox_coverage = float(bbox_area / (img_area + 1e-9))

    blur_var = float(variance_of_laplacian_gray(image))

    feats = {
        "avg_visibility": avg_visibility,
        "shoulder_px": float(shoulder_px),
        "hip_px": float(hip_px),
        "sleeve_px": float(sleeve_px),
        "shoulder_z": shoulder_z,
        "hip_z": hip_z,
        "bbox_coverage": bbox_coverage,
        "blur_var": blur_var,
        "shoulder_to_hip_ratio": float(shoulder_px / (hip_px + 1e-6)),
        "shoulder_to_height_ratio": float(shoulder_px / (height_px + 1e-6)),
        "sleeve_to_height_ratio": float(sleeve_px / (height_px + 1e-6)),
        "sleeve_to_shoulder_ratio": float(sleeve_px / (shoulder_px + 1e-6)),
        "hip_to_height_ratio": float(hip_px / (height_px + 1e-6)),
        "torso_to_height_ratio": float(torso_px / (height_px + 1e-6)),
        "torso_to_shoulder_ratio": float(torso_px / (shoulder_px + 1e-6)),
    }
    return feats
