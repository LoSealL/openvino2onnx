"""
Copyright Wenyi Tang 2024

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Construct a subgraph for DetectionOutput
"""

import io
import warnings

import onnx
import torch
import torch.nn as nn

from .utils import text_to_boolean


class DetectionOutput(nn.Module):
    """DetectionOutput as a torch module to export to ONNX."""

    def __init__(self, attrs, loc_shape, priors_shape, class_shape, out_shape):
        super().__init__()
        normalized = text_to_boolean(attrs.get("normalized", "0"))
        self.prior_size = 4 if normalized else 5
        self.offset = 0 if normalized else 1
        self.num_priors = priors_shape[2] // self.prior_size
        if attrs["version"] == "opset0":
            self.num_classes = attrs["num_classes"]
        else:
            self.num_classes = class_shape[1] // self.num_priors
        self.num_loc_classes = (
            1 if text_to_boolean(attrs.get("share_location", "1")) else self.num_classes
        )
        self.background_id = int(attrs.get("background_label_id", 0))
        self.variance_encoded_in_target = text_to_boolean(
            attrs.get("variance_encoded_in_target")
        )
        self.code_type_corner = "CORNER" in attrs.get("code_type", "")
        if self.num_loc_classes > 1:
            warnings.warn(
                "Multi classes is not tested yet, "
                "if you find any bugs, please submit an issue."
            )

    def get_loc_predictions(self, location):
        """Get location to prediction bboxes

        Args:
            location (Tensor): A tensor of shape [B, N * 4]

        Returns:
            Tensor: A tensor of shape [B, N // C, C, 4]
        """
        B = location.shape[0]
        N = self.num_priors // self.num_loc_classes
        return location.reshape([B, N, self.num_loc_classes, 4])

    def get_confidence_score(self, confidence):
        """Get confidence data to scores

        Args:
            confidence (Tensor): A tensor of shape [B, N * C]

        Returns:
            Tensor: A tensor of shape [B, N, C]
        """
        B = confidence.shape[0]
        N = self.num_priors
        C = self.num_classes
        scores = confidence.reshape([B, N, C])
        if self.num_loc_classes == 1:
            return scores[..., :1]
        return scores

    def get_prior_bboxes(self, prior_data):
        """Get anchers

        Args:
            prior_data (Tensor): A tensor of shape [B, 2, N]

        Returns:
            Tuple[Tensor, Tensor]: A tuple of two tensors, the 1st one is prior boxes
                and the 2nd one is prior variances.
        """
        prior_data = prior_data.reshape([-1, self.num_priors, self.prior_size])
        prior_data = prior_data[..., -4:]
        if not self.variance_encoded_in_target:
            prior_data = prior_data.reshape([-1, 2, self.num_priors, 4])
            return prior_data[:, 0], prior_data[:, 1]
        return prior_data, None

    def decode_bboxes(self, location, anchers, variances):
        anchers = anchers.reshape([1, -1, 1, 4])
        variances = variances.reshape([1, -1, 1, 4])
        if self.code_type_corner:
            bboxes = anchers + variances * location
        else:
            prior_wh = anchers[..., 2:] - anchers[..., :2]
            prior_center = (anchers[..., 2:] + anchers[..., :2]) * 0.5
            center = location[..., :2] * variances[..., :2] * prior_wh + prior_center
            size = torch.exp(variances[..., 2:] * location[..., 2:]) * prior_wh
            bboxes = torch.cat([center - size / 2, center + size / 2], -1)
        return bboxes

    def forward(self, box_logits, proposals, class_preds):
        location = self.get_loc_predictions(box_logits)
        scores = self.get_confidence_score(class_preds)
        anchers, variances = self.get_prior_bboxes(proposals)
        bboxes = self.decode_bboxes(location, anchers, variances)
        return bboxes, scores

    def export(self, loc_shape, prior_shape, class_shape):
        buf = io.BytesIO()
        args = [torch.zeros(shape) for shape in (loc_shape, prior_shape, class_shape)]
        torch.onnx.export(
            self,
            tuple(args),
            buf,
            input_names=["locations", "priors", "confidences"],
            output_names=["bboxes", "scores"],
            opset_version=13,
        )
        buf.seek(0)
        return onnx.load_model(buf)
