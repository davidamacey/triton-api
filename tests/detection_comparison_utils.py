"""
Detection Comparison Utilities

Utilities for comparing object detection outputs from different models/pipelines.
Includes IoU calculation, detection matching, and metrics computation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single object detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    def box(self) -> np.ndarray:
        """Return box as [x1, y1, x2, y2] array."""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def area(self) -> float:
        """Calculate box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    # Calculate intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_iou_matrix(boxes1: List[Detection], boxes2: List[Detection]) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of boxes.

    Args:
        boxes1: List of detections from first source
        boxes2: List of detections from second source

    Returns:
        IoU matrix [N, M] where N=len(boxes1), M=len(boxes2)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    iou_matrix = np.zeros((len(boxes1), len(boxes2)))

    for i, det1 in enumerate(boxes1):
        for j, det2 in enumerate(boxes2):
            iou_matrix[i, j] = calculate_iou(det1.box(), det2.box())

    return iou_matrix


def match_detections(
    reference: List[Detection],
    test: List[Detection],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match detections between reference and test sets using IoU.

    Uses greedy matching: for each reference detection, find the best matching
    test detection above the IoU threshold.

    Args:
        reference: Reference detections (ground truth or baseline)
        test: Test detections to compare
        iou_threshold: Minimum IoU for a match

    Returns:
        - matches: List of (ref_idx, test_idx) pairs
        - unmatched_ref: Indices of unmatched reference detections
        - unmatched_test: Indices of unmatched test detections
    """
    if len(reference) == 0:
        return [], [], list(range(len(test)))
    if len(test) == 0:
        return [], list(range(len(reference))), []

    # Calculate IoU matrix
    iou_matrix = calculate_iou_matrix(reference, test)

    matches = []
    matched_ref = set()
    matched_test = set()

    # Greedy matching: for each reference, find best test match
    for ref_idx in range(len(reference)):
        # Find best matching test detection
        best_iou = 0.0
        best_test_idx = -1

        for test_idx in range(len(test)):
            if test_idx in matched_test:
                continue

            iou = iou_matrix[ref_idx, test_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_test_idx = test_idx

        if best_test_idx >= 0:
            matches.append((ref_idx, best_test_idx))
            matched_ref.add(ref_idx)
            matched_test.add(best_test_idx)

    # Find unmatched detections
    unmatched_ref = [i for i in range(len(reference)) if i not in matched_ref]
    unmatched_test = [i for i in range(len(test)) if i not in matched_test]

    return matches, unmatched_ref, unmatched_test


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two detection sets."""
    num_matches: int
    num_reference: int
    num_test: int
    precision: float  # matched / test
    recall: float     # matched / reference
    f1_score: float
    mean_iou: float   # Average IoU of matched pairs
    mean_conf_diff: float  # Average confidence difference
    mean_box_diff: float   # Average L2 distance between box centers


def calculate_comparison_metrics(
    reference: List[Detection],
    test: List[Detection],
    iou_threshold: float = 0.5
) -> ComparisonMetrics:
    """
    Calculate comprehensive comparison metrics.

    Args:
        reference: Reference detections (baseline)
        test: Test detections
        iou_threshold: Minimum IoU for matching

    Returns:
        ComparisonMetrics object
    """
    matches, unmatched_ref, unmatched_test = match_detections(
        reference, test, iou_threshold
    )

    num_matches = len(matches)
    num_reference = len(reference)
    num_test = len(test)

    # Precision and recall
    precision = num_matches / num_test if num_test > 0 else 0.0
    recall = num_matches / num_reference if num_reference > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate additional metrics for matched pairs
    if num_matches > 0:
        ious = []
        conf_diffs = []
        box_diffs = []

        for ref_idx, test_idx in matches:
            ref_det = reference[ref_idx]
            test_det = test[test_idx]

            # IoU
            iou = calculate_iou(ref_det.box(), test_det.box())
            ious.append(iou)

            # Confidence difference
            conf_diff = abs(ref_det.confidence - test_det.confidence)
            conf_diffs.append(conf_diff)

            # Box center distance
            ref_center = np.array([
                (ref_det.x1 + ref_det.x2) / 2,
                (ref_det.y1 + ref_det.y2) / 2
            ])
            test_center = np.array([
                (test_det.x1 + test_det.x2) / 2,
                (test_det.y1 + test_det.y2) / 2
            ])
            box_diff = np.linalg.norm(ref_center - test_center)
            box_diffs.append(box_diff)

        mean_iou = np.mean(ious)
        mean_conf_diff = np.mean(conf_diffs)
        mean_box_diff = np.mean(box_diffs)
    else:
        mean_iou = 0.0
        mean_conf_diff = 0.0
        mean_box_diff = 0.0

    return ComparisonMetrics(
        num_matches=num_matches,
        num_reference=num_reference,
        num_test=num_test,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        mean_iou=mean_iou,
        mean_conf_diff=mean_conf_diff,
        mean_box_diff=mean_box_diff
    )


def parse_detections(detections: List[Dict]) -> List[Detection]:
    """
    Parse detection dictionaries into Detection objects.

    Args:
        detections: List of detection dicts with keys: x1, y1, x2, y2, confidence, class

    Returns:
        List of Detection objects
    """
    return [
        Detection(
            x1=det['x1'],
            y1=det['y1'],
            x2=det['x2'],
            y2=det['y2'],
            confidence=det['confidence'],
            class_id=det['class']
        )
        for det in detections
    ]


def format_metrics_table(metrics_dict: Dict[str, ComparisonMetrics]) -> str:
    """
    Format comparison metrics as a nice table.

    Args:
        metrics_dict: Dict mapping test name to ComparisonMetrics

    Returns:
        Formatted table string
    """
    header = f"{'Method':<30} {'Prec':<8} {'Recall':<8} {'F1':<8} {'IoU':<8} {'ConfΔ':<8} {'BoxΔ':<8} {'Match':<10}"
    lines = [header, "=" * len(header)]

    for method_name, metrics in metrics_dict.items():
        line = (
            f"{method_name:<30} "
            f"{metrics.precision:>7.3f} "
            f"{metrics.recall:>7.3f} "
            f"{metrics.f1_score:>7.3f} "
            f"{metrics.mean_iou:>7.3f} "
            f"{metrics.mean_conf_diff:>7.3f} "
            f"{metrics.mean_box_diff:>7.1f} "
            f"{metrics.num_matches:>3}/{metrics.num_reference:<3}"
        )
        lines.append(line)

    return "\n".join(lines)
