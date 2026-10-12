import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from infer import FaceLandmarker

# WFLW has no forehead points, so two virtual ones are appended after the 98
# real landmarks (index 98 = left forehead, 99 = right forehead). They sit above
# the outer eyebrow corners, pushed along the chin->eyes direction so they follow
# head roll. FOREHEAD_RISE is a fraction of the eye-to-chin distance; tune it so
# hats / crowns sit where you want.
FOREHEAD_RISE = 0.40
FOREHEAD_LEFT, FOREHEAD_RIGHT = 98, 99


def mesh(
    frame: np.ndarray,
    landmarker: "FaceLandmarker"
) -> Optional[np.ndarray]:
    """
    Detect facial landmarks in a frame using YOLO + landmark CNN.

    Args:
        frame: BGR image frame.
        landmarker: FaceLandmarker instance (from infer.py).

    Returns:
        A (100, 2) array of landmarks in frame pixel coordinates
        (98 WFLW points + 2 virtual forehead points) if a face is
        detected, otherwise None.
    """
    pts = landmarker.get_landmarks(frame)

    if pts is None:
        return None

    eye_mid = (pts[96] + pts[97]) / 2          # between the pupils
    up = eye_mid - pts[16]                     # chin -> eyes
    forehead = np.stack([
        pts[33] + up * FOREHEAD_RISE,          # left outer eyebrow corner
        pts[46] + up * FOREHEAD_RISE           # right outer eyebrow corner
    ])

    return np.vstack([pts, forehead])


def distance(
    p1: tuple[int, int],
    p2: tuple[int, int]
) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First 2D point (x, y).
        p2: Second 2D point (x, y).

    Returns:
        Distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def rotation_angle(
    p1: tuple[int, int],
    p2: tuple[int, int]
) -> float:
    """
    Compute the angle between two points in degrees.

    Args:
        p1: First 2D point (x, y).
        p2: Second 2D point (x, y).

    Returns:
        Rotation angle in degrees.
    """
    return np.rad2deg(
        np.arctan2(
            p1[1] - p2[1],
            p1[0] - p2[0]
        )
    )


def rotate_and_resize(
    image: np.ndarray,
    angle: float,
    scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate and scale an image while preserving all pixels.

    Args:
        image: Input RGBA overlay image.
        angle: Rotation angle in degrees.
        scale: Scaling factor.

    Returns:
        A tuple containing:
            - The transformed image.
            - The affine transformation matrix.
    """
    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    transform_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    cos_theta = abs(transform_matrix[0, 0])
    sin_theta = abs(transform_matrix[0, 1])

    new_w = int(h * sin_theta + w * cos_theta)
    new_h = int(w * sin_theta + h * cos_theta)

    transform_matrix[0, 2] += (new_w / 2) - center[0]
    transform_matrix[1, 2] += (new_h / 2) - center[1]

    transformed_image = cv2.warpAffine(
        image,
        transform_matrix,
        (new_w, new_h)
    )

    return transformed_image, transform_matrix


def paste_overlay(
    frame: np.ndarray,
    overlay_type: str,
    overlay_image: np.ndarray,
    landmarker: "FaceLandmarker",
    overlay_anchors: tuple[
        tuple[int, int],
        tuple[int, int]
    ]
) -> np.ndarray:
    """
    Paste a transformed overlay onto a detected face.

    The overlay is automatically scaled, rotated and
    positioned according to facial landmarks.

    Args:
        frame: Input BGR video frame.
        overlay_type: Overlay category name.
        overlay_image: RGBA overlay image.
        landmarker: FaceLandmarker (YOLO + landmark CNN).
        overlay_anchors: Overlay reference anchor points.

    Returns:
        Frame with the overlay applied.
    """
    global FACE_ANCHORS

    left_idx, right_idx = FACE_ANCHORS[overlay_type]

    overlay_anchor_left, overlay_anchor_right = overlay_anchors

    frame_h, frame_w, _ = frame.shape

    landmarks = mesh(frame, landmarker)

    if landmarks is None:
        return frame[:,:,::-1]

    overlay_reference_width = distance(
        overlay_anchor_left,
        overlay_anchor_right
    )

    # landmarks are already in frame pixels (no * frame_w / * frame_h)
    left_face_point = (
        int(landmarks[left_idx][0]),
        int(landmarks[left_idx][1])
    )

    right_face_point = (
        int(landmarks[right_idx][0]),
        int(landmarks[right_idx][1])
    )

    scale_factor = (
        distance(left_face_point, right_face_point)
        / overlay_reference_width
    )

    angle = rotation_angle(
        right_face_point,
        left_face_point
    )

    transformed_overlay, transform_matrix = rotate_and_resize(
        overlay_image,
        -angle,
        scale_factor
    )

    overlay_h, overlay_w = transformed_overlay.shape[:2]

    transformed_anchor_left = np.int64(
        transform_matrix @ np.array([
            [overlay_anchor_left[0]],
            [overlay_anchor_left[1]],
            [1]
        ])
    )

    transformed_anchor_left = (
        transformed_anchor_left[0, 0],
        transformed_anchor_left[1, 0]
    )

    paste_top_left = (
        left_face_point[0] - transformed_anchor_left[0],
        left_face_point[1] - transformed_anchor_left[1]
    )

    paste_bottom_right = (
        paste_top_left[0] + overlay_w,
        paste_top_left[1] + overlay_h
    )

    crop_left = 0
    crop_top = 0
    crop_right = 1
    crop_bottom = 1

    if paste_top_left[0] < 0:
        crop_left = -paste_top_left[0]

    if paste_top_left[1] < 0:
        crop_top = -paste_top_left[1]

    if paste_bottom_right[0] > frame_w:
        crop_right = paste_bottom_right[0] - frame_w

    if paste_bottom_right[1] > frame_h:
        crop_bottom = paste_bottom_right[1] - frame_h

    transformed_overlay = transformed_overlay[
        crop_top:-crop_bottom,
        crop_left:-crop_right
    ]

    paste_top_left = (
        paste_top_left[0] + crop_left,
        paste_top_left[1] + crop_top
    )

    overlay_w -= crop_left + crop_right
    overlay_h -= crop_top + crop_bottom
    frame = frame[:,:,::-1]
    roi = frame[
        paste_top_left[1]:paste_top_left[1] + overlay_h,
        paste_top_left[0]:paste_top_left[0] + overlay_w
    ]

    alpha_mask = (
        transformed_overlay[:, :, 3] == 255
    )[:, :, np.newaxis]

    frame[
        paste_top_left[1]:paste_top_left[1] + overlay_h,
        paste_top_left[0]:paste_top_left[0] + overlay_w
    ] = np.where(
        alpha_mask,
        transformed_overlay[:, :, :3],
        roi
    )

    return frame


# (left, right) anchor indices into the 100-point array from mesh():
# 0-97 are WFLW landmarks, 98/99 are the virtual forehead points.
# "left" = left side of the image, same convention as before.
FACE_ANCHORS: dict[str, tuple[int, int]] = {
    "glasses": (60, 72),         # outer eye corners
    "moustache": (76, 82),       # mouth corners
    "beard": (10, 22),           # lower jaw (approximate, tune)
    "mask": (1, 31),             # face edge at cheek level (approximate, tune)
    "crown": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "hat": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "bunny_ears": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "devil_horns": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "halo": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "pirate_hat": (FOREHEAD_LEFT, FOREHEAD_RIGHT),
    "mouth": (76, 82),
    "eye_patch": (60, 64),       # outer + inner corner of the left-in-image eye
    "left_eye": (60, 64),
    "right_eye": (68, 72),       # inner + outer corner of the right-in-image eye
    "nose": (55, 59),            # nostril outer edges
    "clown_nose": (55, 59)
}
