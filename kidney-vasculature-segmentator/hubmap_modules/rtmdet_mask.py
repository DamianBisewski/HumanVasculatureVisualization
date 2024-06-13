import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.ops import roi_align
from mmdet.registry import MODELS
from mmdet.models.detectors import RTMDet
from mmdet.models.utils import unpack_gt_instances
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask


@MODELS.register_module()
class RTMDetWithMaskHead(RTMDet):

    def __init__(self, mask_head, *args, norm_eval=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_head = MODELS.build(mask_head)
        assert self.train_cfg.mask_pos_mode in (
            'all',
            'weighted_sum',
        )
        self.strides = [
            stride[0] for stride in self.bbox_head.prior_generator.strides
        ]
        if self.train_cfg.mask_pos_mode == 'weighted_sum':
            self.fpn_weight = nn.Parameter(
                torch.ones(len(self.strides), dtype=torch.float32),
                requires_grad=True,
            )
            self.fpn_weight_relu = nn.ReLU()
            self.eps = 1e-6
        self.norm_eval = norm_eval


    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def loss(self, batch_inputs, batch_data_samples):
        img_feats = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(img_feats, batch_data_samples)

        strides = self.strides
        mask_feat_size = self.train_cfg.mask_roi_size
        batch_gt_instances = unpack_gt_instances(batch_data_samples)[0]

        num_bboxes = sum(
            img_gt_instances.bboxes.size(0)
            for img_gt_instances in batch_gt_instances)
        if num_bboxes == 0:
            losses['loss_mask'] = sum(
                self.mask_head(feat).sum() for feat in batch_inputs) * 0
            return losses

        gt_masks = []
        gt_bboxes = []
        for img_gt_instances in batch_gt_instances:
            img_masks = img_gt_instances.masks
            img_bboxes = img_gt_instances.bboxes
            img_gt_roi_masks = img_masks  # cropped in data pipeline
            gt_masks.append(img_gt_roi_masks)
            gt_bboxes.append(img_bboxes)

        gt_masks = torch.cat(gt_masks)
        mask_feats = []
        mask_gt = []
        for stride, feat in zip(strides, img_feats):
            roi_feats = roi_align(feat,
                                  gt_bboxes,
                                  mask_feat_size,
                                  spatial_scale=1 / stride,
                                  aligned=True)
            mask_feats.append(roi_feats)
            mask_gt.append(gt_masks)

        if self.train_cfg.mask_pos_mode == 'all':
            mask_feats = torch.cat(mask_feats)
            mask_gt = torch.cat(mask_gt)
        else:  # weighted sum
            weight = self.fpn_weight_relu(self.fpn_weight)
            weight = weight / (weight.sum() + self.eps)
            mask_feats = sum(mask_feat * w
                             for mask_feat, w in zip(mask_feats, weight))
            # mask_feats = sum(mask_feats)
            mask_gt = mask_gt[0]
        mask_pred = self.mask_head(mask_feats).squeeze(1)

        losses['loss_mask'] = F.binary_cross_entropy_with_logits(
            mask_pred, mask_gt)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        img_feats = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(img_feats,
                                              batch_data_samples,
                                              rescale=False)

        for img_idx, (results,
                      img_meta) in enumerate(zip(results_list,
                                                 batch_img_metas)):
            pred_bboxes = results.bboxes
            mask_feats = []
            for stride, feat in zip(self.strides, img_feats):
                roi_feats = roi_align(feat, [pred_bboxes],
                                      self.train_cfg.mask_roi_size,
                                      spatial_scale=1 / stride,
                                      aligned=True)
                mask_feats.append(roi_feats)
            if self.train_cfg.mask_pos_mode == 'all':
                mask_feats = torch.cat(mask_feats)
            else:  # weighted sum
                weight = self.fpn_weight_relu(self.fpn_weight)
                weight = weight / (weight.sum() + self.eps)
                mask_feats = sum(mask_feat * w
                                 for mask_feat, w in zip(mask_feats, weight))
            mask_pred = self.mask_head(mask_feats)

            # rescale bboxes
            assert img_meta.get('scale_factor') is not None
            results.bboxes /= results.bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

            img_h, img_w = img_meta['ori_shape'][:2]
            # mask_prob = mask_pred.sigmoid()
            # mask_conf = (mask_prob - 0.5).abs() * 2
            # mask_conf = mask_conf.mean((1, 2, 3))
            # results.scores *= mask_conf
            img_mask = _do_paste_mask(mask_pred, pred_bboxes, img_h, img_w, False)[0] > 0

            results.masks = img_mask

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
