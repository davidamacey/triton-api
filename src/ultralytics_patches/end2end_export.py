"""
Ultralytics End2End Export Patch
==================================

Adds TensorRT EfficientNMS plugin support for end-to-end YOLO export.
This enables ONNX models with GPU-accelerated NMS baked directly into the graph.

Source: https://github.com/levipereira/ultralytics
Version: Based on ultralytics v8.3.18
License: AGPL-3.0

Usage:
    from ultralytics_patches import apply_end2end_patch
    apply_end2end_patch()  # Apply once before using YOLO

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.export(
        format="onnx_trt",
        topk_all=300,
        iou_thres=0.7,
        conf_thres=0.25,
        dynamic=True,
        half=True,
        normalize_boxes=True  # Output boxes in [0, 1] range (recommended for Track E)
    )

Custom Arguments:
    normalize_boxes (bool): If True, output boxes are normalized to [0, 1] range.
        Default: False (outputs pixel coordinates in 640x640 space).
        When True, boxes can be scaled to any image size by multiplying:
        box_pixels = box_normalized * [img_w, img_h, img_w, img_h]
"""

import os
import torch
import torch.nn as nn

__version__ = '1.0.0'
__author__ = 'Levi Pereira (original), Extracted for monkey-patch'


# ============================================================================
# TensorRT Custom Operators - EfficientNMS Plugin
# ============================================================================


class TRT_EfficientNMS_85(torch.autograd.Function):
    """TensorRT EfficientNMS operation for TensorRT 8.5+"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
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
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
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
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT_EfficientNMS(torch.autograd.Function):
    """TensorRT EfficientNMS operation with class-agnostic support (TensorRT 8.6+)"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        batch_size, num_boxes, num_classes = scores.shape
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
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
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


class TRT_EfficientNMSX_85(torch.autograd.Function):
    """TensorRT EfficientNMS with indices output (for segmentation)"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            'TRT::EfficientNMSX_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices


class TRT_EfficientNMSX(torch.autograd.Function):
    """TensorRT EfficientNMS with indices and class-agnostic support"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):
        out = g.op(
            'TRT::EfficientNMSX_TRT',
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
            outputs=5,
        )
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices


class TRT_ROIAlign(torch.autograd.Function):
    """TensorRT ROIAlign operation (for instance segmentation masks)"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,  # 1- avg pooling  / 0 - max pooling
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        return g.op(
            'TRT::ROIAlign_TRT',
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )


# ============================================================================
# ONNX Wrapper Modules
# ============================================================================


class ONNX_EfficientNMS_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS operation for detection models"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
        input_size=640,
        normalize_boxes=False,
    ):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes
        # Normalization: output boxes in [0, 1] range instead of pixel coordinates
        # This makes downstream processing simpler - just multiply by any target image size
        self.input_size = float(input_size)
        self.normalize_boxes = normalize_boxes

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]  # YOLO11 main output is first element
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )

        # Normalize boxes to [0, 1] range if requested
        # This makes downstream processing trivial: box_pixels = box_normalized * [img_w, img_h, img_w, img_h]
        # Matches Ultralytics Boxes.xyxyn property behavior
        if self.normalize_boxes:
            det_boxes = det_boxes / self.input_size

        return num_det, det_boxes, det_scores, det_classes


class ONNX_EfficientNMSX_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS operation (with indices for segmentation)"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
    ):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]  # YOLO11 main output is first element
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )
        return num_det, det_boxes, det_scores, det_classes, det_indices


class ONNX_End2End_MASK_TRT(torch.nn.Module):
    """ONNX module with TensorRT NMS and ROIAlign for instance segmentation"""

    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=160,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80,
    ):
        super().__init__()
        assert isinstance(max_wh, (int)) or max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.max_obj = max_obj
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, x):
        det = x[0]
        proto = x[1]
        det = det.permute(0, 2, 1)

        bboxes_x = det[..., 0:1]
        bboxes_y = det[..., 1:2]
        bboxes_w = det[..., 2:3]
        bboxes_h = det[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=-1)
        bboxes = bboxes.unsqueeze(2)  # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        scores = det[..., 4 : 4 + self.n_classes]

        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = det[..., 4 + self.n_classes : 4 + self.n_classes + nm]

        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
                self.class_agnostic,
            )
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(
                bboxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )

        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]

        pooled_proto = TRT_ROIAlign.apply(
            proto,
            det_boxes.view(total_object, 4),
            batch_indices,
            1,
            1,
            self.mask_resolution,
            self.mask_resolution,
            self.sampling_ratio,
            self.pooler_scale,
        )
        pooled_proto = pooled_proto.view(
            total_object,
            nm,
            self.mask_resolution * self.mask_resolution,
        )

        det_masks = (
            torch.matmul(det_masks.unsqueeze(dim=1), pooled_proto)
            .sigmoid()
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
        )

        return num_det, det_boxes, det_scores, det_classes, det_masks


class End2End_TRT(torch.nn.Module):
    """Wrapper module for end-to-end ONNX/TensorRT export with NMS"""

    def __init__(
        self,
        model,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=56,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80,
        is_det_model=True,
        v10detect=False,
        input_size=640,
        normalize_boxes=False,
    ):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.v10detect = v10detect

        if is_det_model and not self.v10detect:
            self.model.model[-1].end2end = False
            self.patch_model = ONNX_EfficientNMS_TRT
            self.end2end = self.patch_model(
                class_agnostic,
                max_obj,
                iou_thres,
                score_thres,
                max_wh,
                device,
                n_classes,
                input_size=input_size,
                normalize_boxes=normalize_boxes,
            )
            self.end2end.eval()
        elif not is_det_model and not self.v10detect:
            self.model.model[-1].end2end = False
            self.patch_model = ONNX_End2End_MASK_TRT
            self.end2end = self.patch_model(
                class_agnostic,
                max_obj,
                iou_thres,
                score_thres,
                mask_resolution,
                pooler_scale,
                sampling_ratio,
                max_wh,
                device,
                n_classes,
            )
            self.end2end.eval()
        elif self.v10detect:
            self.model.model[-1].end2end = True

    def forward(self, x):
        if not self.v10detect:
            # For YOLOv8/YOLOv11, use the end2end process
            x = self.model(x)
            x = self.end2end(x)
            return x
        else:
            # For YOLOv10, manually handle the detection outputs
            x = self.model(x)
            det_boxes = x[:, :, :4]
            det_scores = x[:, :, 4]
            det_classes = x[:, :, 5].int()
            num_dets = (x[:, :, 4] > 0.0).sum(dim=1, keepdim=True).int()
            return num_dets, det_boxes, det_scores, det_classes


# ============================================================================
# Export Method - Monkey Patch Target
# ============================================================================


def export_onnx_trt(self, prefix='ONNX TRT:'):
    """
    Export YOLO model to ONNX format with TensorRT EfficientNMS plugin.

    This method wraps the model with End2End_TRT to bake NMS into the ONNX graph.

    Args:
        prefix (str): Logging prefix

    Returns:
        tuple: (export_path, onnx_model)
    """
    try:
        from ultralytics.utils import colorstr, LOGGER
        from ultralytics.utils.checks import check_requirements
    except ImportError:
        # Fallback if imports fail
        def colorstr(s):
            return s

        import logging

        LOGGER = logging.getLogger(__name__)

        def check_requirements(reqs):
            pass

    requirements = ['onnx>=1.12.0']

    if self.args.simplify:
        requirements += [
            'onnxsim>=0.4.33',
            'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime',
        ]
    check_requirements(requirements)

    import onnx  # noqa

    # Detect model type
    try:
        from ultralytics.nn.modules import v10Detect
    except ImportError:
        # Create dummy v10Detect if not available
        class v10Detect:
            pass

    try:
        from ultralytics.models.yolo.model import SegmentationModel
    except ImportError:
        # Create dummy SegmentationModel if not available
        class SegmentationModel:
            pass

    labels = len(self.model.names)
    is_det_model = True
    v10detect = False

    for k, m in self.model.named_modules():
        if isinstance(m, v10Detect):
            v10detect = True
            break

    # Save label file
    if len(self.model.names.keys()) > 0:
        label_file = os.path.splitext(self.file)[0] + '-trt.txt'
        with open(label_file, 'w') as f_trt:
            for name in self.model.names.values():
                f_trt.write(name + '\n')
        LOGGER.info(f"{prefix} Successfully generated the label file: '{label_file}'.")

    # Get opset version
    try:
        from ultralytics.engine.exporter import get_latest_opset

        opset_version = self.args.opset or get_latest_opset()
    except (ImportError, AttributeError):
        opset_version = self.args.opset or 17

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...')

    f = os.path.splitext(self.file)[0] + '-trt.onnx'

    batch_size = 'batch'
    dynamic = self.args.dynamic
    dynamic_axes = {
        'images': {0: 'batch', 2: 'height', 3: 'width'},
    }  # variable length axes
    output_axes = {
        'num_dets': {0: 'batch'},
        'det_boxes': {0: 'batch'},
        'det_scores': {0: 'batch'},
        'det_classes': {0: 'batch'},
    }

    d = {
        'stride': int(max(self.model.stride)),
        'names': self.model.names,
        'model type': 'Segmentation' if isinstance(self.model, SegmentationModel) else 'Detection',
        'train input': f'{tuple(self.im.shape[1:])} - CHW',
        'TRT Compatibility': '8.6 or above' if self.args.class_agnostic else '8.5 or above',
    }
    if not v10detect:
        d['TRT Plugins'] = (
            'TRT_EfficientNMSX, ROIAlign'
            if isinstance(self.model, SegmentationModel)
            else 'TRT_EfficientNMS'
        )

    if not isinstance(self.model, SegmentationModel):
        is_det_model = True
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        shapes = [
            batch_size,
            1,
            batch_size,
            self.args.topk_all,
            4,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
        ]

    else:
        is_det_model = False
        output_axes['det_masks'] = {0: 'batch'}
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes', 'det_masks']
        shapes = [
            batch_size,
            1,
            batch_size,
            self.args.topk_all,
            4,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
            batch_size,
            self.args.topk_all,
            self.args.mask_resolution * self.args.mask_resolution,
        ]

    dynamic_axes.update(output_axes)

    if v10detect:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.float()
        self.model.fuse()
        for k, m in self.model.named_modules():
            if isinstance(m, v10Detect):
                m.max_det = self.args.topk_all

    # Get input size for box normalization
    input_size = int(self.im.shape[-1])  # Last dim is width (assumes square input)
    normalize_boxes = getattr(self.args, 'normalize_boxes', False)

    # Wrap model with End2End_TRT
    if v10detect:
        self.model = nn.Sequential(
            self.model,
            End2End_TRT(
                self.model,
                self.args.class_agnostic,
                self.args.topk_all,
                self.args.iou_thres,
                self.args.conf_thres,
                self.args.mask_resolution,
                self.args.pooler_scale,
                self.args.sampling_ratio,
                None,
                self.args.device,
                labels,
                is_det_model,
                v10detect,
                input_size=input_size,
                normalize_boxes=normalize_boxes,
            ),
        )
    else:
        self.model = End2End_TRT(
            self.model,
            self.args.class_agnostic,
            self.args.topk_all,
            self.args.iou_thres,
            self.args.conf_thres,
            self.args.mask_resolution,
            self.args.pooler_scale,
            self.args.sampling_ratio,
            None,
            self.args.device,
            labels,
            is_det_model,
            v10detect,
            input_size=input_size,
            normalize_boxes=normalize_boxes,
        )

    # Export to ONNX
    torch.onnx.export(
        self.model.cpu() if dynamic else self.model,  # dynamic=True only compatible with cpu
        self.im.cpu() if dynamic else self.im,
        f,
        verbose=False,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Force legacy ONNX exporter (new torch.export fails with End2End models)
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Add metadata
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    # Set output shapes
    for i in model_onnx.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    # Simplify
    check_requirements('onnxsim')
    try:
        import onnxsim

        LOGGER.info(f'\n{prefix} Starting to simplify ONNX...')
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
    except Exception as e:
        LOGGER.info(f'\n{prefix} Simplifier failure: {e}')

    onnx.save(model_onnx, f)

    # Cleanup with onnx_graphsurgeon
    check_requirements('onnx_graphsurgeon')

    LOGGER.info(f'\n{prefix} Starting to cleanup ONNX using onnx_graphsurgeon...')
    try:
        import onnx_graphsurgeon as gs

        graph = gs.import_onnx(model_onnx)
        graph = graph.cleanup().toposort()
        model_onnx = gs.export_onnx(graph)
        onnx.save(model_onnx, f)
    except Exception as e:
        LOGGER.info(f'\n{prefix} Cleanup failure: {e}')

    LOGGER.info(f'{prefix} export success ✅, saved as {f}')
    return f, model_onnx


# ============================================================================
# Monkey Patch Application
# ============================================================================

_patch_applied = False


def apply_end2end_patch():
    """
    Apply the end2end export patch to ultralytics.

    This adds the export_onnx_trt() method to the Exporter class and
    patches the __init__ to add custom End2End arguments.

    Usage:
        from ultralytics_patches import apply_end2end_patch
        apply_end2end_patch()

        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        model.export(format="onnx_trt", ...)
    """
    global _patch_applied

    if _patch_applied:
        print('⚠️  End2End patch already applied, skipping...')
        return

    try:
        from ultralytics.engine.exporter import Exporter
    except ImportError as e:
        raise ImportError(f'Failed to import ultralytics.engine.exporter: {e}')

    # Save original __init__
    original_init = Exporter.__init__

    # Patched __init__ that adds End2End arguments
    def patched_init(self, cfg=None, overrides=None, _callbacks=None):
        # Call original __init__ - only pass cfg if not None to avoid ultralytics 8.3+ compatibility issues
        if cfg is not None:
            original_init(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        else:
            original_init(self, overrides=overrides, _callbacks=_callbacks)

        # Add End2End custom arguments with defaults if not present
        if not hasattr(self.args, 'topk_all'):
            self.args.topk_all = 300
        if not hasattr(self.args, 'iou_thres'):
            self.args.iou_thres = 0.7
        if not hasattr(self.args, 'conf_thres'):
            self.args.conf_thres = 0.25
        if not hasattr(self.args, 'class_agnostic'):
            self.args.class_agnostic = False
        if not hasattr(self.args, 'mask_resolution'):
            self.args.mask_resolution = 56
        if not hasattr(self.args, 'pooler_scale'):
            self.args.pooler_scale = 0.25
        if not hasattr(self.args, 'sampling_ratio'):
            self.args.sampling_ratio = 0
        if not hasattr(self.args, 'normalize_boxes'):
            self.args.normalize_boxes = False  # Default: pixel coordinates (640x640)

    # Apply patches
    Exporter.__init__ = patched_init
    Exporter.export_onnx_trt = export_onnx_trt

    # Patch export format list
    try:
        from ultralytics.engine import exporter

        if hasattr(exporter, 'export_formats'):
            # Add onnx_trt to supported formats
            formats = exporter.export_formats()
            if 'ONNX TensorRT' not in formats['Format'].values:
                import pandas as pd

                new_row = pd.DataFrame(
                    [
                        {
                            'Format': 'ONNX TensorRT',
                            'Argument': 'onnx_trt',
                            'Suffix': '_trt.onnx',
                            'CPU': True,
                            'GPU': True,
                        }
                    ]
                )
                exporter.export_formats = lambda: pd.concat([formats, new_row], ignore_index=True)
    except Exception as e:
        print(f'⚠️  Could not update export_formats table: {e}')

    _patch_applied = True
    print('✅ End2End TensorRT NMS patch applied successfully!')
    print("   You can now use: model.export(format='onnx_trt', ...)")


def is_patch_applied():
    """Check if the patch has been applied"""
    return _patch_applied


__all__ = [
    'apply_end2end_patch',
    'is_patch_applied',
    'export_onnx_trt',
    'End2End_TRT',
    'ONNX_EfficientNMS_TRT',
    'ONNX_EfficientNMSX_TRT',
    'ONNX_End2End_MASK_TRT',
    'TRT_EfficientNMS',
    'TRT_EfficientNMS_85',
    'TRT_EfficientNMSX',
    'TRT_EfficientNMSX_85',
    'TRT_ROIAlign',
]
