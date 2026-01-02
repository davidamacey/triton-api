"""
Face Alignment Utility for ArcFace Recognition

Production-grade face alignment using techniques from:
- InsightFace (Alibaba): Standard reference landmarks, umeyama algorithm
- DeepFace (Facebook/Meta): Similarity transform estimation
- FaceNet (Google): Centered alignment for embedding extraction

Pipeline:
1. SCRFD outputs 5 facial landmarks per face
2. Compute similarity transform using Umeyama algorithm (industry standard)
3. Apply affine warp to align face to 112x112

Reference landmarks are calibrated for the 112x112 input size used by
ArcFace, CosFace, and other modern face recognition models.

References:
- InsightFace: https://github.com/deepinsight/insightface
- Umeyama algorithm: "Least-squares estimation of transformation parameters"
"""

import numpy as np


# =============================================================================
# ArcFace Reference Landmarks (112x112 aligned face)
# =============================================================================

# Standard reference landmarks for ArcFace face recognition
# These positions are optimized for the 112x112 input size
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # Left eye center
        [73.5318, 51.5014],  # Right eye center
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ],
    dtype=np.float32,
)

# Output size for aligned faces
ARCFACE_INPUT_SIZE = 112


# =============================================================================
# Umeyama Algorithm (Industry Standard)
# =============================================================================
# Used by InsightFace, DeepFace, and other production systems
# Reference: "Least-squares estimation of transformation parameters between
#            two point patterns" by Shinji Umeyama, 1991


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
) -> np.ndarray:
    """
    Estimate N-D similarity transformation with or without scaling.

    This is the Umeyama algorithm - the industry standard for face alignment
    used by InsightFace (Alibaba), DeepFace (Meta), and others.

    Parameters:
        src: Source points [N, D] - detected landmarks
        dst: Destination points [N, D] - reference landmarks
        estimate_scale: Whether to estimate scaling (True for face alignment)

    Returns:
        T: (D+1, D+1) homogeneous transformation matrix

    The transformation is defined as:
        T = [s*R | t]
            [0   | 1]

    Where:
        s: Scale factor (if estimate_scale=True)
        R: Rotation matrix
        t: Translation vector
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Compute covariance matrix
    A = dst_demean.T @ src_demean / num

    # Compute SVD
    U, S, Vt = np.linalg.svd(A)

    # Construct rotation matrix with proper handling of reflection
    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    # Rotation
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    # Scale
    if estimate_scale:
        src_var = src_demean.var(axis=0).sum()
        scale = 1.0 / src_var * (S @ d)
    else:
        scale = 1.0

    T[:dim, :dim] *= scale
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)

    return T


def estimate_similarity_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Estimate 2D similarity transform using Umeyama algorithm.

    This is the standard method used by InsightFace for face alignment.

    Args:
        src_pts: Source points [N, 2] - detected landmarks
        dst_pts: Destination points [N, 2] - reference landmarks

    Returns:
        M: 2x3 affine transformation matrix for cv2.warpAffine
    """
    # Use Umeyama algorithm
    T = umeyama(src_pts.astype(np.float64), dst_pts.astype(np.float64), True)

    # Extract 2x3 affine matrix
    return T[:2, :].astype(np.float32)


def get_alignment_matrix(
    landmarks: np.ndarray,
    output_size: int = ARCFACE_INPUT_SIZE,
) -> np.ndarray:
    """
    Compute alignment matrix to transform face to standard position.

    Args:
        landmarks: Detected facial landmarks [5, 2] from SCRFD
                  Order: left_eye, right_eye, nose, left_mouth, right_mouth
        output_size: Target output size (default: 112 for ArcFace)

    Returns:
        M: 2x3 affine transformation matrix
    """
    # Scale reference landmarks to output size
    scale = output_size / 112.0
    ref_landmarks = ARCFACE_REF_LANDMARKS * scale

    # Estimate similarity transform
    return estimate_similarity_transform(landmarks, ref_landmarks)


def get_inverse_alignment_matrix(M: np.ndarray) -> np.ndarray:
    """
    Compute inverse of alignment matrix for mapping back to original coordinates.

    Args:
        M: 2x3 affine transformation matrix

    Returns:
        M_inv: 2x3 inverse affine matrix
    """
    # Add [0, 0, 1] row to make it 3x3
    M_full = np.vstack([M, [0, 0, 1]])

    # Invert
    M_inv_full = np.linalg.inv(M_full)

    # Return 2x3
    return M_inv_full[:2, :].astype(np.float32)


# =============================================================================
# Face Alignment Functions
# =============================================================================


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = ARCFACE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align a face image using detected landmarks.

    Args:
        image: Input image [H, W, C] or [H, W] (numpy array)
        landmarks: Facial landmarks [5, 2] from SCRFD
        output_size: Target output size (default: 112)

    Returns:
        aligned: Aligned face image [output_size, output_size, C]
        M: Transformation matrix used
    """
    import cv2

    # Get alignment matrix
    M = get_alignment_matrix(landmarks, output_size)

    # Apply affine warp
    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return aligned, M


def align_faces_batch(
    image: np.ndarray,
    landmarks_batch: np.ndarray,
    output_size: int = ARCFACE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align multiple faces from a single image.

    Args:
        image: Input image [H, W, C]
        landmarks_batch: Landmarks for multiple faces [N, 5, 2]
        output_size: Target output size

    Returns:
        aligned_faces: Aligned face images [N, output_size, output_size, C]
        matrices: Transformation matrices [N, 2, 3]
    """

    n_faces = landmarks_batch.shape[0]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    aligned_faces = np.zeros(
        (n_faces, output_size, output_size, channels),
        dtype=image.dtype,
    )
    matrices = np.zeros((n_faces, 2, 3), dtype=np.float32)

    for i in range(n_faces):
        aligned, M = align_face(image, landmarks_batch[i], output_size)
        if len(aligned.shape) == 2:
            aligned = aligned[:, :, np.newaxis]
        aligned_faces[i] = aligned
        matrices[i] = M

    return aligned_faces, matrices


# =============================================================================
# Preprocessing for ArcFace
# =============================================================================


def preprocess_for_arcface(
    aligned_face: np.ndarray,
) -> np.ndarray:
    """
    Preprocess aligned face for ArcFace inference.

    ArcFace expects: [B, 3, 112, 112] FP32, normalized as (x - 127.5) / 128.0

    Args:
        aligned_face: Aligned face image [112, 112, 3] uint8 RGB

    Returns:
        tensor: Preprocessed tensor [1, 3, 112, 112] FP32
    """
    # Convert to float and normalize
    face = aligned_face.astype(np.float32)
    face = (face - 127.5) / 128.0

    # Transpose HWC -> CHW
    face = np.transpose(face, (2, 0, 1))

    # Add batch dimension
    return np.expand_dims(face, axis=0)


def preprocess_faces_batch(
    aligned_faces: np.ndarray,
) -> np.ndarray:
    """
    Preprocess batch of aligned faces for ArcFace inference.

    Args:
        aligned_faces: Aligned face images [N, 112, 112, 3] uint8 RGB

    Returns:
        tensor: Preprocessed tensors [N, 3, 112, 112] FP32
    """
    # Convert to float and normalize
    faces = aligned_faces.astype(np.float32)
    faces = (faces - 127.5) / 128.0

    # Transpose NHWC -> NCHW
    return np.transpose(faces, (0, 3, 1, 2))


# =============================================================================
# Landmark Conversion
# =============================================================================


def scrfd_landmarks_to_points(
    landmarks_flat: np.ndarray,
) -> np.ndarray:
    """
    Convert SCRFD flat landmark output to point array.

    SCRFD outputs landmarks as [N, 10] with format:
    [lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lx5, ly5]

    Args:
        landmarks_flat: Flat landmarks [N, 10] or [10]

    Returns:
        points: Landmark points [N, 5, 2] or [5, 2]
    """
    if landmarks_flat.ndim == 1:
        return landmarks_flat.reshape(5, 2)
    return landmarks_flat.reshape(-1, 5, 2)


def scale_landmarks_to_original(
    landmarks: np.ndarray,
    original_size: tuple[int, int],
    input_size: int = 640,
) -> np.ndarray:
    """
    Scale landmarks from SCRFD input size to original image coordinates.

    Args:
        landmarks: Landmarks in SCRFD input space [N, 5, 2] or [5, 2]
        original_size: Original image size (height, width)
        input_size: SCRFD input size (default: 640)

    Returns:
        scaled: Landmarks in original image coordinates
    """
    orig_h, orig_w = original_size

    # Calculate scale factors (assuming letterbox padding)
    scale = min(input_size / orig_w, input_size / orig_h)
    pad_w = (input_size - orig_w * scale) / 2
    pad_h = (input_size - orig_h * scale) / 2

    # Remove padding and scale back
    scaled = landmarks.copy().astype(np.float32)

    if scaled.ndim == 2:
        scaled[:, 0] = (scaled[:, 0] - pad_w) / scale
        scaled[:, 1] = (scaled[:, 1] - pad_h) / scale
    else:
        scaled[:, :, 0] = (scaled[:, :, 0] - pad_w) / scale
        scaled[:, :, 1] = (scaled[:, :, 1] - pad_h) / scale

    return scaled


# =============================================================================
# Quality Assessment
# =============================================================================


def compute_face_quality(
    landmarks: np.ndarray,
    box: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """
    Compute face quality score based on landmarks and box.

    Quality factors:
    - Face size relative to image
    - Landmark symmetry (frontal face detection)
    - Face within image bounds

    Args:
        landmarks: Facial landmarks [5, 2]
        box: Face bounding box [x1, y1, x2, y2]
        image_size: Image dimensions (height, width)

    Returns:
        quality: Quality score 0.0 to 1.0
    """
    h, w = image_size
    x1, y1, x2, y2 = box

    # Face size score (larger is better, up to a point)
    face_area = (x2 - x1) * (y2 - y1)
    image_area = h * w
    size_ratio = face_area / image_area
    size_score = min(1.0, size_ratio * 10)  # Cap at 10% of image

    # Boundary score (penalize faces at edge)
    margin = 0.02
    boundary_score = 1.0
    if x1 < w * margin or x2 > w * (1 - margin):
        boundary_score *= 0.8
    if y1 < h * margin or y2 > h * (1 - margin):
        boundary_score *= 0.8

    # Symmetry score (eyes should be roughly level)
    left_eye, right_eye = landmarks[0], landmarks[1]
    eye_tilt = abs(left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0] + 1e-6)
    symmetry_score = max(0, 1 - eye_tilt * 2)

    # Combined quality
    quality = size_score * boundary_score * symmetry_score

    return float(np.clip(quality, 0, 1))
