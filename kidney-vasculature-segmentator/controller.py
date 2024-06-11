# controller.py
import os
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image
import cv2
import numpy as np

from model import SegmentationAppModel
from view import SegmentationAppView

class SegmentationAppController:
    """
    Controller class for the Medical Image Segmentation Application.
    Handles user interactions, manages the model and view.
    """

    def __init__(self, root, detectors, masks_detector_index, device='cuda:0'):
        """
        Initializes the controller with the root window, detectors, and device.

        Args:
            root (Tk): The root window.
            detectors (list): List of tuples containing config file, checkpoint file, and description.
            masks_detector_index (int): Index of the special model used for mask detection.
            device (device): Device to be used for model inference (default: 'cuda:0').
        """
        self.model = SegmentationAppModel(detectors, masks_detector_index, device)
        self.view = SegmentationAppView(root, self)
        self.image_files = []
        self.current_index = 0
        self.original_image = None
        self.gt_original_image = None
        self.combined_results = None
        self.image_path = None

    def change_font_size(self):
        """Prompt the user to change the font size of the UI elements."""
        new_font_size = simpledialog.askinteger("Font Size", "Enter new font size:", initialvalue=self.view.font_size)
        if new_font_size:
            self.view.update_font_size(new_font_size)

    def load_image(self):
        """Load a single image file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.tif")])
        if file_path:
            self.clear_metrics()
            self.display_image(file_path)

    def load_directory(self):
        """Load all image files from a selected directory."""
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
                                file.endswith(('.jpg', '.png', '.tif'))]
            self.current_index = 0
            if self.image_files:
                self.clear_metrics()
                self.display_image(self.image_files[self.current_index])
            else:
                messagebox.showwarning("Warning", "No image files found in the selected directory")

    def next_image(self):
        """Load the next image in the directory."""
        if hasattr(self, 'image_files') and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.clear_metrics()
            self.display_image(self.image_files[self.current_index])

    def prev_image(self):
        """Load the previous image in the directory."""
        if hasattr(self, 'image_files') and self.current_index > 0:
            self.current_index -= 1
            self.clear_metrics()
            self.display_image(self.image_files[self.current_index])

    def clear_metrics(self):
        """Clear displayed IoU and mAP metrics."""
        self.view.display_ious([])
        self.view.display_map(None)

    def display_image(self, file_path):
        """
        Display an image on the left canvas.

        Args:
            file_path (str): Path to the image file.
        """
        self.image_path = file_path
        self.original_image = self.load_image_with_consistent_color(file_path)
        self.update_image('left')
        self.display_meta_info(file_path)
        self.display_ground_truth(file_path)

    def load_image_with_consistent_color(self, file_path):
        """
        Load an image with consistent color processing.

        Args:
            file_path (str): Path to the image file.

        Returns:
            Image: Loaded image.
        """
        return Image.fromarray(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))

    def display_meta_info(self, file_path):
        """
        Display metadata information for the current image.

        Args:
            file_path (str): Path to the image file.
        """
        if self.model.tile_meta is not None:
            base_name = os.path.basename(file_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            row = self.model.tile_meta[self.model.tile_meta['id'] == base_name_no_ext]
            if not row.empty:
                info = row.to_string(index=False)
                self.view.update_meta_info(info)

    def update_image(self, canvas='left'):
        """
        Update the image displayed on the specified canvas.

        Args:
            canvas (str): Canvas to update ('left' or 'right').
        """
        if canvas == 'left' and self.original_image is not None:
            image = self.original_image.copy()
            width, height = image.size
            new_size = (int(width * self.view.get_zoom_level(canvas)), int(height * self.view.get_zoom_level(canvas)))
            image = image.resize(new_size, Image.LANCZOS)
            self.view.display_image(image, canvas)
        elif canvas == 'right' and self.gt_original_image is not None:
            image = self.gt_original_image.copy()
            width, height = image.size
            new_size = (int(width * self.view.get_zoom_level(canvas)), int(height * self.view.get_zoom_level(canvas)))
            image = image.resize(new_size, Image.LANCZOS)
            self.view.display_image(image, canvas)

    def zoom(self, event, canvas):
        """
        Zoom in or out on the specified canvas.

        Args:
            event (Event): Mouse wheel event.
            canvas (str): Canvas to zoom ('left' or 'right').
        """
        if event.delta > 0 or event.num == 4:
            self.view.zoom_in(canvas)
        elif event.delta < 0 or event.num == 5:
            self.view.zoom_out(canvas)
        self.update_image(canvas)

    def start_pan(self, event, canvas):
        """
        Start panning the image on the specified canvas.

        Args:
            event (Event): Mouse button press event.
            canvas (str): Canvas to pan ('left' or 'right').
        """
        if canvas == 'left':
            self.view.left_canvas.scan_mark(event.x, event.y)
        elif canvas == 'right':
            self.view.right_canvas.scan_mark(event.x, event.y)

    def pan_image(self, event, canvas):
        """
        Pan the image on the specified canvas.

        Args:
            event (Event): Mouse motion event.
            canvas (str): Canvas to pan ('left' or 'right').
        """
        if canvas == 'left':
            self.view.left_canvas.scan_dragto(event.x, event.y, gain=1)
        elif canvas == 'right':
            self.view.right_canvas.scan_dragto(event.x, event.y, gain=1)

    def resize_canvas(self, event):
        """Resize the canvas and update the image."""
        self.update_image('left')
        self.update_image('right')

    def segment_image(self):
        """Segment the loaded image using the selected model and update the results."""
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("Warning", "Please load an image first")
            return

        try:
            if self.view.segmentation_mode.get() == "MMDetection":
                results = self.model.perform_inference_mmdet(self.image_path)
            else:
                config_path = 'configs/r0i.py'
                checkpoint_path = 'checkpoints/r0i.pth'
                results = self.model.perform_inference_sahi(config_path, checkpoint_path,
                                                            self.image_path, self.view.score_threshold.get(),
                                                            self.view.height.get(), self.view.width.get(),
                                                            self.view.overlap_height_ratio.get(),
                                                            self.view.overlap_width_ratio.get(),
                                                            self.model.device)

            self.update_segmented_image(results)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def update_segmented_image(self, combined_results):
        """
        Update the displayed image with segmentation results and calculate metrics.

        Args:
            combined_results (list): Segmentation results.
        """
        self.combined_results = combined_results
        image = self.load_image_with_consistent_color(self.image_path)

        self.original_image = self.model.visualize_combined_results(image, combined_results, self.view.score_threshold.get())
        
        self.view.reset_zoom_level('left')  # Reset zoom level
        self.update_image('left')

        # Calculate IoU and mAP
        base_name_no_ext = os.path.splitext(os.path.basename(self.image_path))[0]
        if base_name_no_ext in self.model.ground_truth_annotations:
            annotations = self.model.ground_truth_annotations[base_name_no_ext]
            ious = self.model.calculate_and_display_ious(combined_results, annotations, self.view.score_threshold.get())
            map_score = self.model.calculate_map(combined_results, annotations, self.view.score_threshold.get())
            self.view.display_ious(ious)
            self.view.display_map(map_score)
        else:
            self.view.display_ious([])
            self.view.display_map(None)

        # Update current configuration label
        self.view.lbl_current_config.config(
            text=f"Current Config: {self.view.segmentation_mode.get()}, Score Threshold: {self.view.score_threshold.get()}"
        )

    def load_results(self):
        """Load segmentation results from a .npy file."""
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if file_path:
            self.loaded_results = np.load(file_path, allow_pickle=True).item()
            self.image_path = self.loaded_results['image_path']
            self.combined_results = self.loaded_results['combined_results']
            self.model.class_names = self.loaded_results['class_names']

            image = self.load_image_with_consistent_color(self.image_path)
            self.original_image = self.model.visualize_combined_results(image, self.combined_results,
                                                                        self.view.score_threshold.get())
            self.update_image('left')
            messagebox.showinfo("Load Results", "Results loaded successfully")

    def save_results(self):
        """Save segmentation results to a .npy file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy files", "*.npy")])
        if file_path:
            results = {
                'image_path': self.image_path,
                'combined_results': self.combined_results,
                'class_names': self.model.class_names,
                'zoom_levels': self.view.zoom_levels  # Save zoom levels
            }
            np.save(file_path, results)
            messagebox.showinfo("Save Results", "Results saved successfully")

    def load_tile_meta(self):
        """Load tile metadata from a CSV file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.model.load_tile_meta(file_path)
            messagebox.showinfo("Load tile_meta.csv", "tile_meta.csv loaded successfully")

    def load_annotations_jsonl(self):
        """Load annotations from a JSON Lines file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSON Lines files", "*.jsonl")])
        if file_path:
            self.model.load_annotations_jsonl(file_path)
            messagebox.showinfo("Load Annotations JSONL", "Annotations JSONL loaded successfully")

    def display_ground_truth(self, file_path):
        """
        Display ground truth annotations on the right canvas.

        Args:
            file_path (str): Path to the image file.
        """
        base_name = os.path.basename(file_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        if base_name_no_ext in self.model.ground_truth_annotations:
            annotations = self.model.ground_truth_annotations[base_name_no_ext]
            image = self.load_image_with_consistent_color(file_path)
            ground_truth_image_pil = self.model.visualize_ground_truth(image, annotations)
            self.gt_original_image = ground_truth_image_pil  # Store ground truth image
            self.update_image('right')
        else:
            self.view.right_canvas.delete("all")
            self.gt_original_image = None
