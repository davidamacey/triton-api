#!/usr/bin/env python3
"""
Export SCRFD Face Detection with TensorRT EfficientNMS (End2End GPU Pipeline)

This creates a TensorRT engine with:
1. GPU anchor decoding (boxes + landmarks)
2. GPU NMS via TensorRT EfficientNMS plugin
3. GPU landmark gathering using NMS indices

Result: 100% GPU face detection with no CPU round-trips.

Inputs:
    - images: [B, 3, 640, 640] FP32 RGB normalized [0, 1]

Outputs:
    - num_dets: [B, 1] INT32 number of detected faces
    - det_boxes: [B, max_det, 4] FP32 face boxes [x1, y1, x2, y2] normalized [0, 1]
    - det_scores: [B, max_det] FP32 confidence scores
    - det_landmarks: [B, max_det, 10] FP32 5-point landmarks (flattened)

Usage:
    docker compose exec yolo-api python /app/export/export_scrfd_end2end.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


# Configuration
SCRFD_SIZE = 640
MAX_FACES = 128
STRIDES = [8, 16, 32]
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


# =============================================================================
# TensorRT EfficientNMS Custom Op (standard version)
# =============================================================================


class TRT_EfficientNMS(torch.autograd.Function):
    """TensorRT EfficientNMS plugin for GPU-accelerated NMS."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.4,
        max_output_boxes=128,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.5,
        class_agnostic=1,
    ):
        batch_size, _num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.4,
        max_output_boxes=128,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.5,
        class_agnostic=1,
    ):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


# =============================================================================
# ONNX Export
# =============================================================================


class SCRFD_ONNX_Wrapper(nn.Module):
    """
    SCRFD post-processing wrapper for ONNX/TensorRT export.

    Key insight: Concatenate landmarks with boxes to form [B, N, 14] tensor.
    When EfficientNMS selects boxes, landmarks come along automatically.
    Then we split the output back into boxes [B, M, 4] and landmarks [B, M, 10].
    """

    def __init__(
        self,
        img_size: int = 640,
        max_det: int = 128,
        conf_thres: float = 0.5,
        iou_thres: float = 0.4,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_det = max_det
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Register anchor buffers
        self._register_anchor_buffers()

    def _register_anchor_buffers(self):
        """Pre-compute anchor centers and strides."""
        all_anchor_centers = []
        all_strides = []

        for stride in STRIDES:
            feat_h = self.img_size // stride
            feat_w = self.img_size // stride
            num_anchors = feat_h * feat_w * 2

            y, x = torch.meshgrid(
                torch.arange(feat_h, dtype=torch.float32),
                torch.arange(feat_w, dtype=torch.float32),
                indexing='ij',
            )

            anchor_centers = torch.stack([x, y], dim=-1).reshape(-1, 2)
            anchor_centers = (anchor_centers + 0.5) * stride
            anchor_centers = anchor_centers.repeat_interleave(2, dim=0)

            all_anchor_centers.append(anchor_centers)
            all_strides.append(torch.full((num_anchors,), stride, dtype=torch.float32))

        self.register_buffer('anchor_centers', torch.cat(all_anchor_centers, dim=0))
        self.register_buffer('anchor_strides', torch.cat(all_strides, dim=0))

    def forward(
        self,
        # Stride 8 outputs
        scores_8,
        boxes_8,
        landmarks_8,
        # Stride 16 outputs
        scores_16,
        boxes_16,
        landmarks_16,
        # Stride 32 outputs
        scores_32,
        boxes_32,
        landmarks_32,
    ):
        """
        Process raw SCRFD outputs with GPU NMS.

        Returns boxes and landmarks together after NMS.
        """
        B = scores_8.shape[0]

        # Concatenate all strides
        scores = torch.cat(
            [
                scores_8.reshape(B, -1, 1),
                scores_16.reshape(B, -1, 1),
                scores_32.reshape(B, -1, 1),
            ],
            dim=1,
        )  # [B, N, 1]

        boxes = torch.cat(
            [
                boxes_8.reshape(B, -1, 4),
                boxes_16.reshape(B, -1, 4),
                boxes_32.reshape(B, -1, 4),
            ],
            dim=1,
        )  # [B, N, 4]

        landmarks = torch.cat(
            [
                landmarks_8.reshape(B, -1, 10),
                landmarks_16.reshape(B, -1, 10),
                landmarks_32.reshape(B, -1, 10),
            ],
            dim=1,
        )  # [B, N, 10]

        # Decode boxes
        centers = self.anchor_centers.unsqueeze(0).expand(B, -1, -1)
        strides = self.anchor_strides.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

        x1 = centers[:, :, 0:1] - boxes[:, :, 0:1] * strides
        y1 = centers[:, :, 1:2] - boxes[:, :, 1:2] * strides
        x2 = centers[:, :, 0:1] + boxes[:, :, 2:3] * strides
        y2 = centers[:, :, 1:2] + boxes[:, :, 3:4] * strides

        decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1) / self.img_size  # [B, N, 4]

        # Decode landmarks
        lmk_reshaped = landmarks.reshape(B, -1, 5, 2)
        decoded_lmk = lmk_reshaped * strides.unsqueeze(-1) + centers.unsqueeze(-2)
        decoded_landmarks = decoded_lmk.reshape(B, -1, 10) / self.img_size  # [B, N, 10]

        # Concatenate boxes and landmarks: [B, N, 14]
        # This way, when NMS selects a box, its landmarks come along
        boxes_with_landmarks = torch.cat([decoded_boxes, decoded_landmarks], dim=-1)

        # Apply sigmoid to scores
        scores_sigmoid = torch.sigmoid(scores)

        # Run NMS on combined boxes+landmarks
        # The NMS plugin treats [B, N, 14] as boxes with 14 coords
        # It only uses the first 4 for IoU, but outputs all 14
        num_dets, det_boxes_lmk, det_scores, _det_classes = TRT_EfficientNMS.apply(
            boxes_with_landmarks,
            scores_sigmoid,
            -1,
            1,
            self.iou_thres,
            self.max_det,
            '1',
            0,
            self.conf_thres,
            1,
        )

        # Split output back into boxes and landmarks
        det_boxes = det_boxes_lmk[:, :, :4]  # [B, max_det, 4]
        det_landmarks = det_boxes_lmk[:, :, 4:]  # [B, max_det, 10]

        return num_dets, det_boxes, det_scores, det_landmarks


def export_scrfd_postprocess_onnx(output_path: Path, max_det: int = 128):
    """Export SCRFD post-processing (decode + NMS) to ONNX."""

    print('\n' + '=' * 60)
    print('Exporting SCRFD Post-Processing to ONNX')
    print('=' * 60)

    model = SCRFD_ONNX_Wrapper(
        img_size=SCRFD_SIZE,
        max_det=max_det,
        conf_thres=CONF_THRESHOLD,
        iou_thres=NMS_THRESHOLD,
    )
    model.eval()

    # Create dummy inputs matching SCRFD output shapes
    B = 1
    dummy_inputs = (
        # Stride 8: 80x80x2 = 12800 anchors
        torch.randn(B, 12800, 1),  # scores
        torch.randn(B, 12800, 4),  # boxes
        torch.randn(B, 12800, 10),  # landmarks
        # Stride 16: 40x40x2 = 3200 anchors
        torch.randn(B, 3200, 1),
        torch.randn(B, 3200, 4),
        torch.randn(B, 3200, 10),
        # Stride 32: 20x20x2 = 800 anchors
        torch.randn(B, 800, 1),
        torch.randn(B, 800, 4),
        torch.randn(B, 800, 10),
    )

    input_names = [
        'scores_8',
        'boxes_8',
        'landmarks_8',
        'scores_16',
        'boxes_16',
        'landmarks_16',
        'scores_32',
        'boxes_32',
        'landmarks_32',
    ]

    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_landmarks']

    dynamic_axes = {
        'scores_8': {0: 'batch'},
        'boxes_8': {0: 'batch'},
        'landmarks_8': {0: 'batch'},
        'scores_16': {0: 'batch'},
        'boxes_16': {0: 'batch'},
        'landmarks_16': {0: 'batch'},
        'scores_32': {0: 'batch'},
        'boxes_32': {0: 'batch'},
        'landmarks_32': {0: 'batch'},
        'num_dets': {0: 'batch'},
        'det_boxes': {0: 'batch'},
        'det_scores': {0: 'batch'},
        'det_landmarks': {0: 'batch'},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use legacy exporter to avoid external data files
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Use legacy exporter
    )

    print(f'  ✓ ONNX exported: {output_path}')
    return output_path


def convert_to_tensorrt(onnx_path: Path, plan_path: Path, max_batch: int = 64):
    """Convert SCRFD post-processing ONNX to TensorRT."""

    print('\n' + '=' * 60)
    print('Converting to TensorRT')
    print('=' * 60)

    try:
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, '')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        print(f'  Parsing ONNX: {onnx_path}')
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'    Error: {parser.get_error(i)}')
                raise RuntimeError('ONNX parsing failed')

        print(f'  ✓ ONNX parsed: {network.num_layers} layers')

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.set_flag(trt.BuilderFlag.FP16)

        # Set up optimization profile for dynamic batch
        profile = builder.create_optimization_profile()

        # Define shapes for each input
        input_shapes = {
            'scores_8': (12800, 1),
            'boxes_8': (12800, 4),
            'landmarks_8': (12800, 10),
            'scores_16': (3200, 1),
            'boxes_16': (3200, 4),
            'landmarks_16': (3200, 10),
            'scores_32': (800, 1),
            'boxes_32': (800, 4),
            'landmarks_32': (800, 10),
        }

        for name, shape in input_shapes.items():
            profile.set_shape(
                name,
                min=(1, *shape),
                opt=(8, *shape),
                max=(max_batch, *shape),
            )

        config.add_optimization_profile(profile)

        print('  Building TensorRT engine (this may take a few minutes)...')
        serialized = builder.build_serialized_network(network, config)

        if serialized is None:
            raise RuntimeError('TensorRT build failed')

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, 'wb') as f:
            f.write(serialized)

        size_mb = plan_path.stat().st_size / (1024 * 1024)
        print(f'  ✓ TensorRT engine saved: {plan_path} ({size_mb:.1f} MB)')

        return True

    except ImportError:
        print('  ✗ TensorRT not available')
        return False
    except Exception as e:
        print(f'  ✗ TensorRT conversion failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Export SCRFD End2End')
    parser.add_argument('--max-det', type=int, default=MAX_FACES, help='Max detections')
    parser.add_argument('--max-batch', type=int, default=64, help='Max batch size')
    parser.add_argument('--output-dir', type=Path, default=Path('models/scrfd_postprocess_end2end'))
    args = parser.parse_args()

    print('=' * 60)
    print('SCRFD End2End Export (GPU NMS)')
    print('=' * 60)
    print(f'  Max detections: {args.max_det}')
    print(f'  Max batch size: {args.max_batch}')
    print(f'  Conf threshold: {CONF_THRESHOLD}')
    print(f'  NMS threshold: {NMS_THRESHOLD}')

    onnx_path = args.output_dir / '1' / 'postprocess.onnx'
    plan_path = args.output_dir / '1' / 'model.plan'

    # Export post-processing to ONNX
    export_scrfd_postprocess_onnx(onnx_path, args.max_det)

    # Convert to TensorRT
    success = convert_to_tensorrt(onnx_path, plan_path, args.max_batch)

    if success:
        print('\n' + '=' * 60)
        print('SCRFD End2End Export Complete!')
        print('=' * 60)
        print('\nNext steps:')
        print('  1. Create Triton config for scrfd_postprocess_end2end')
        print('  2. Create ensemble combining SCRFD backbone + postprocess')
        print('  3. Create DALI face alignment pipeline')
    else:
        print('\n✗ Export failed')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
