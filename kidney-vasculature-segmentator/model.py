import os
import json
import mmcv
from mmdet.apis import inference_detector, init_detector
import numpy as np
import pandas as pd
import cv2
import torch
from ensemble_boxes import weighted_boxes_fusion
from shapely.geometry import Polygon
from mmdet.evaluation.functional.mean_ap import eval_map

class SegmentationAppModel:
    """
    Model class for the Medical Image Segmentation Application.
    Handles loading models, performing inference, and calculating metrics.
    """

    def __init__(self, detectors, special_model_index, device='cuda:0'):
        """
        Initializes the model with the given detectors and device.

        Args:
            detectors (list): List of tuples containing config file, checkpoint file, and description.
            special_model_index (int): Index of the special model used for mask detection.
            device (str): Device to be used for model inference (default: 'cuda:0').
        """
        self.device = device if device else self.get_default_device()
        self.models, self.masks_detector = self.load_models(detectors, special_model_index, self.device)
        self.class_names = ['blood_vessel', 'glomerulus', 'unsure']
        self.tile_meta = None
        self.ground_truth_annotations = {}

    def get_default_device(self):
        """Return the default device to use for tensors."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self, detectors, special_model_index, device='cuda:0'):
        """
        Load the models for inference.

        Args:
            detectors (list): List of tuples containing config file, checkpoint file, and description.
            special_model_index (int): Index of the special model used for mask detection.
            device (str): Device to be used for model inference.

        Returns:
            tuple: Loaded models and the special mask detector model.
        """
        models = []
        special_model = None
        for i, (config_file, checkpoint_file, config_description) in enumerate(detectors):
            model = init_detector(config_file, checkpoint_file, device=device)
            models.append(model)
            if i == special_model_index:
                special_model = model
        return models, special_model

    def generate_img_id(self, image_path):
        """
        Generate a unique identifier for the image based on its file name.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Unique identifier for the image.
        """
        return os.path.splitext(os.path.basename(image_path))[0]

    @torch.no_grad()
    def predict_mask(self, result, input_size=(1440, 1440)):
        """
        Predict masks using the special mask detector model.

        Args:
            result (dict): Detection result containing image path and predicted instances.
            input_size (tuple): Size to which the image should be resized for mask prediction.

        Returns:
            dict: Prediction result including predicted masks.
        """
        from mmengine.structures.instance_data import InstanceData
        from mmdet.structures import DetDataSample
        from PIL import Image

        image = Image.open(result['img_path']).convert('RGB')
        img = np.array(image.resize(input_size))

        # Convert the image to a torch tensor and move it to the GPU
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        batch_data = dict(
            inputs=img_tensor,
            data_samples=[
                DetDataSample(metainfo=dict(img_id=result['img_id'],
                                            ori_shape=(512, 512),
                                            img_shape=(1440, 1440),
                                            img_path=result['img_path'],
                                            scale_factor=(1440 / 512, 1440 / 512)))
            ])
        batch_data = self.masks_detector.data_preprocessor(batch_data, False)
        batch_data_inputs = batch_data['inputs']
        batch_data_samples = batch_data['data_samples']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        img_feats = self.masks_detector.extract_feat(batch_data_inputs)

        img_result = InstanceData()
        for k, v in result['pred_instances'].items():
            img_result[k] = v.to(self.device)
        img_result.bboxes *= 1440 / 512
        results_list = self.masks_detector.roi_head.predict_mask(img_feats,
                                                            batch_img_metas, [img_result],
                                                            rescale=True)
        out = results_list[0].cpu()
        ret = dict(img_id=result['img_id'],
                   ori_shape=(512, 512),
                   img_shape=(1440, 1440),
                   img_path=result['img_path'],
                   scale_factor=(1440 / 512, 1440 / 512))
        ret['pred_instances'] = (out['bboxes'].numpy(), out['scores'].numpy(),
                                 out['labels'].numpy(), out['masks'].numpy(),)
        return ret

    def perform_inference_mmdet(self, image_path):
        """
        Perform inference using MMDetection models and predict masks.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list: Unified inference results including bounding boxes, scores, labels, and masks.
        """
        results = []
        for model in self.models:
            result = inference_detector(model, image_path)
            results.append(result)
        weights = [
            2, 2, 2, 2, 1, 1, 1, 1, 2, 2
        ]

        SCALER = 10000
        IOU_THR = 0.7

        boxes_list = [(r.pred_instances.bboxes / SCALER).tolist() for r in results]
        scores_list = [r.pred_instances.scores.tolist() for r in results]
        labels_list = [r.pred_instances.labels.tolist() for r in results]

        boxes, scores, labels = weighted_boxes_fusion(boxes_list,
                                                      scores_list,
                                                      labels_list,
                                                      weights=weights,
                                                      iou_thr=IOU_THR,
                                                      conf_type='avg')
        pred_instances = dict(
            bboxes=torch.from_numpy(boxes).float() * SCALER,
            scores=torch.from_numpy(scores).float(),
            labels=torch.from_numpy(labels).long(),
        )

        combined_pred_result = {'pred_instances': pred_instances,
                                'img_path': image_path,
                                'img_id': 0
                                }
        return self.unify_results(self.predict_mask(combined_pred_result))

    def perform_inference_sahi(self, config_path, checkpoint_path,
                               image_path, confidence_threshold,
                               slice_height, slice_width,
                               overlap_height_ratio,
                               overlap_width_ratio, device):
        """
        Perform inference using the SAHI model with slicing.

        Args:
            config_path (str): Path to the config file.
            checkpoint_path (str): Path to the checkpoint file.
            image_path (str): Path to the image file.
            confidence_threshold (float): Confidence threshold for predictions.
            slice_height (int): Height of the slices.
            slice_width (int): Width of the slices.
            overlap_height_ratio (float): Overlap ratio for height.
            overlap_width_ratio (float): Overlap ratio for width.
            device (str): Device to be used for model inference.

        Returns:
            list: Unified inference results including bounding boxes, scores, labels, and masks.
        """
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='mmdet',
            model_path=checkpoint_path,
            config_path=config_path,
            confidence_threshold=confidence_threshold,
            category_mapping={'0': 'blood vessel',
                              '1': 'glomerulus',
                              '2': 'unsure'},
            device=device
        )

        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        return self.unify_results_sahi(result)

    def unify_results(self, result):
        """
        Unify the results from MMDetection into a standard format.

        Args:
            result (dict): Prediction result including bounding boxes, scores, labels, and masks.

        Returns:
            list: Unified results.
        """
        bboxes, scores, labels, masks = result['pred_instances']
        unified_results = []
        for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
            unified_results.append({
                'bbox': bbox,
                'score': score,
                'label': label,
                'mask': mask
            })
        return unified_results

    def unify_results_sahi(self, result):
        """
        Unify the results from SAHI into a standard format.

        Args:
            result (SlicedPredictionResult): SAHI prediction result.

        Returns:
            list: Unified results.
        """
        unified_results = []
        for pred in result.object_prediction_list:
            bbox = [pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy]
            unified_results.append({
                'bbox': bbox,
                'score': pred.score.value,
                'label': pred.category.id,
                'mask': pred.mask.bool_mask if pred.mask else None
            })
        return unified_results

    def visualize_combined_results(self, image, combined_results, score_threshold=0.01):
        """
        Visualize the combined results on the image.

        Args:
            image (Image): The original image.
            combined_results (list): Unified inference results.
            score_threshold (float): Score threshold for displaying results.

        Returns:
            Image: Image with visualized results.
        """
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        colors = plt.get_cmap('tab20')

        for result in combined_results:
            if result['score'] > score_threshold:
                bbox = result['bbox']
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1

                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                mask = result['mask']
                if mask is not None:
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 0:  # Ensure contour is not empty
                            contour = contour.squeeze()
                            if contour.ndim == 2 and contour.shape[0] > 1:  # Ensure contour is valid
                                ax.plot(contour[:, 0], contour[:, 1], linewidth=2, color=colors(result['label'] % colors.N))

                ax.text(x1, y1 - 2, f'{self.class_names[result["label"]]} {result["score"]:.2f}', fontsize=12, color='white',
                        bbox=dict(facecolor='red', alpha=0.5))

        ax.axis('off')
        plt.tight_layout()

        # Redraw the canvas to update the renderer
        fig.canvas.draw()

        # Get the renderer and extract the image as RGBA
        img_array = np.array(fig.canvas.get_renderer().buffer_rgba())

        plt.close(fig)
        img_pil = Image.fromarray(img_array)
        return img_pil

    def visualize_ground_truth(self, image, annotations):
        """
        Visualize ground truth annotations on the image.

        Args:
            image (Image): The original image.
            annotations (list): List of ground truth annotations.

        Returns:
            Image: Image with visualized ground truth annotations.
        """
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        colors = plt.get_cmap('tab20')

        for annotation in annotations:
            polygon = np.array(annotation['coordinates'][0])
            label = self.class_names.index(annotation['type'])

            patch = patches.Polygon(polygon, closed=True, edgecolor=colors(label), facecolor=colors(label, alpha=0.3))
            ax.add_patch(patch)

            x, y = polygon[0]
            ax.text(x, y - 2, annotation['type'], fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

        ax.axis('off')
        plt.tight_layout()

        # Redraw the canvas to update the renderer
        fig.canvas.draw()

        # Get the renderer and extract the image as RGBA
        img_array = np.array(fig.canvas.get_renderer().buffer_rgba())

        plt.close(fig)
        img_pil = Image.fromarray(img_array)
        return img_pil

    def load_tile_meta(self, file_path):
        """
        Load tile metadata from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.tile_meta = pd.read_csv(file_path)

    def load_annotations_jsonl(self, file_path):
        """
        Load annotations from a JSON Lines file.

        Args:
            file_path (str): Path to the JSON Lines file.
        """
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.ground_truth_annotations[data['id']] = data['annotations']

    def calculate_iou(self, pred_mask, gt_polygon):
        """
        Calculate the Intersection over Union (IoU) between a predicted mask and a ground truth polygon.

        Args:
            pred_mask (numpy.ndarray): Predicted mask.
            gt_polygon (list): Ground truth polygon coordinates.

        Returns:
            float: IoU score.
        """
        pred_mask = pred_mask.astype(np.uint8)

        # Create binary mask for the ground truth polygon
        gt_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        # Ensure gt_mask is contiguous
        gt_mask = np.ascontiguousarray(gt_mask)

        # Ensure gt_mask is a 2D numpy array compatible with cv::Mat
        if len(gt_mask.shape) == 3 and gt_mask.shape[2] == 1:
            gt_mask = gt_mask[:, :, 0]

        cv2.fillPoly(gt_mask, [np.array(gt_polygon, dtype=np.int32)], 1)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou

    def calculate_and_display_ious(self, combined_results, annotations, confidence_thr):
        """
        Calculate and display IoU scores for the combined results.

        Args:
            combined_results (list): Unified inference results.
            annotations (list): List of ground truth annotations.
            confidence_thr (float): Confidence threshold for predictions.

        Returns:
            list: List of IoU scores.
        """
        ious = []
        for annotation in annotations:
            gt_polygon = annotation['coordinates'][0]
            best_iou = 0
            for result in combined_results:
                score = result['score']
                if score < confidence_thr:
                    continue
                mask = result['mask']
                if mask is not None:
                    iou = self.calculate_iou(mask, gt_polygon)
                    if iou > best_iou:
                        best_iou = iou
            ious.append(best_iou)
        return ious

    def prepare_mmdet_results(self, combined_results, confidence_thr):
        """
        Prepare detection results for mAP calculation.

        Args:
            combined_results (list): Unified inference results.
            confidence_thr (float): Confidence threshold for predictions.

        Returns:
            list: Prepared detection results.
        """
        det_results = [[] for _ in range(len(self.class_names))]
        for result in combined_results:
            score = result['score']
            if score < confidence_thr:
                continue
            bbox = result['bbox']
            label = result['label']
            det_results[label].append(np.hstack((bbox, score)))
        det_results = [np.array(cls_result) if cls_result else np.zeros((0, 5)) for cls_result in det_results]
        return det_results

    def prepare_annotations(self, annotations):
        """
        Prepare ground truth annotations for mAP calculation.

        Args:
            annotations (list): List of ground truth annotations.

        Returns:
            dict: Prepared annotations including bounding boxes and labels.
        """
        bboxes = []
        labels = []
        for annotation in annotations:
            gt_polygon = annotation['coordinates'][0]
            gt_bbox = Polygon(gt_polygon).bounds
            bboxes.append(gt_bbox)
            labels.append(self.class_names.index(annotation['type']))
        return dict(bboxes=np.array(bboxes), labels=np.array(labels))

    def calculate_map(self, combined_results, annotations, confidence_thr):
        """
        Calculate the mean Average Precision (mAP) score.

        Args:
            combined_results (list): Unified inference results.
            annotations (list): List of ground truth annotations.
            confidence_thr (float): Confidence threshold for predictions.

        Returns:
            float: mAP score.
        """
        det_results = [self.prepare_mmdet_results(combined_results, confidence_thr)]
        gt_annotations = [self.prepare_annotations(annotations)]

        if not det_results or not gt_annotations:
            return 0.0

        try:
            mean_ap, _ = eval_map(det_results, gt_annotations, iou_thr=0.5, dataset=self.class_names)
            return mean_ap
        except Exception as e:
            print(f"Error during mAP calculation: {e}")
            return 0.0
