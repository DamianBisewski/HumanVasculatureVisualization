# view.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import ImageTk

class SegmentationAppView:
    """
    View class for the Medical Image Segmentation Application.
    Manages the graphical user interface (GUI).
    """

    def __init__(self, root, controller, detectors):
        """
        Initializes the view with the root window and controller.

        Args:
            root (Tk): The root window.
            controller (SegmentationAppController): The controller instance.
            detectors (list): List of tuples containing config file, checkpoint file, and description.
        """
        self.root = root
        self.controller = controller
        self.font_size = 14  # Default font size
        self.zoom_levels = {'left': 1.0, 'right': 1.0}  # Separate zoom levels for each canvas
        self.tk_image = None  # Initialize to avoid garbage collection issues
        self.tk_gt_image = None  # Initialize to avoid garbage collection issues
        self.images_to_delete = []
        self.detector_mode_options = [detector[3] for detector in detectors]
        
        self.init_gui()

    def init_gui(self):
        """Initialize the graphical user interface (GUI)."""
        self.root.title("Medical Image Segmentation")

        self.menu_bar = tk.Menu(self.root, font=("Helvetica", self.font_size))
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0, font=("Helvetica", self.font_size))
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.controller.load_image)
        file_menu.add_command(label="Load Directory", command=self.controller.load_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Load Results", command=self.controller.load_results)
        file_menu.add_command(label="Save Results", command=self.controller.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Load tile_meta.csv", command=self.controller.load_tile_meta)
        file_menu.add_command(label="Load Annotations JSONL", command=self.controller.load_annotations_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        self.options_menu = tk.Menu(self.menu_bar, tearoff=0, font=("Helvetica", self.font_size))
        self.menu_bar.add_cascade(label="Options", menu=self.options_menu)
        self.options_menu.add_command(label="Change Font Size", command=self.controller.change_font_size)

        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.left_canvas = tk.Canvas(self.frame)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_canvas.bind("<MouseWheel>", lambda event: self.controller.zoom(event, 'left'))
        self.left_canvas.bind("<Button-4>", lambda event: self.controller.zoom(event, 'left'))  # For Linux
        self.left_canvas.bind("<Button-5>", lambda event: self.controller.zoom(event, 'left'))  # For Linux
        self.left_canvas.bind("<B1-Motion>", lambda event: self.controller.pan_image(event, 'left'))  # For panning
        self.left_canvas.bind("<ButtonPress-1>", lambda event: self.controller.start_pan(event, 'left'))  # For panning
        self.left_canvas.bind("<Configure>", self.controller.resize_canvas)  # Handle window resize

        self.right_canvas = tk.Canvas(self.frame)
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_canvas.bind("<MouseWheel>", lambda event: self.controller.zoom(event, 'right'))
        self.right_canvas.bind("<Button-4>", lambda event: self.controller.zoom(event, 'right'))  # For Linux
        self.right_canvas.bind("<Button-5>", lambda event: self.controller.zoom(event, 'right'))  # For Linux
        self.right_canvas.bind("<B1-Motion>", lambda event: self.controller.pan_image(event, 'right'))  # For panning
        self.right_canvas.bind("<ButtonPress-1>", lambda event: self.controller.start_pan(event, 'right'))  # For panning
        self.right_canvas.bind("<Configure>", self.controller.resize_canvas)  # Handle window resize

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(fill=tk.X)

        self.btn_prev_image = tk.Button(self.btn_frame, text="Previous", command=self.controller.prev_image,
                                        font=("Helvetica", self.font_size))
        self.btn_prev_image.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_next_image = tk.Button(self.btn_frame, text="Next", command=self.controller.next_image,
                                        font=("Helvetica", self.font_size))
        self.btn_next_image.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_segment = tk.Button(self.btn_frame, text="Segment", command=self.controller.segment_image,
                                     font=("Helvetica", self.font_size))
        self.btn_segment.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_segmentation_mode = tk.Label(self.btn_frame, text="Segmentation Mode:",
                                              font=("Helvetica", self.font_size))
        self.lbl_segmentation_mode.pack(side=tk.LEFT, padx=5, pady=5)

        self.segmentation_mode = tk.StringVar(value="MMDet ensemble")  # Default segmentation mode
        self.segmentation_mode_dropdown = tk.OptionMenu(self.btn_frame, self.segmentation_mode, "MMDet ensemble",
                                                        "SAHI", *self.detector_mode_options)
        self.segmentation_mode_dropdown.config(font=("Helvetica", self.font_size))
        self.segmentation_mode_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_score_threshold = tk.Label(self.btn_frame, text="Score Threshold:", font=("Helvetica", self.font_size))
        self.lbl_score_threshold.pack(side=tk.LEFT, padx=5, pady=5)

        self.score_threshold = tk.DoubleVar(value=0.3)
        self.score_threshold_spinbox = tk.Spinbox(self.btn_frame, from_=0.0, to=1.0, increment=0.1,
                                                  textvariable=self.score_threshold, font=("Helvetica", self.font_size))
        self.score_threshold_spinbox.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_current_config = tk.Label(self.btn_frame,
                                           text=f"Current Config: {self.segmentation_mode.get()}, Score Threshold: {self.score_threshold.get()}",
                                           font=("Helvetica", self.font_size))
        self.lbl_current_config.pack(side=tk.LEFT, padx=5, pady=5)

        self.meta_info_text = tk.Text(self.root, height=2, font=("Helvetica", self.font_size))
        self.meta_info_text.pack(fill=tk.X, padx=10, pady=10)
        self.meta_info_text.config(state=tk.DISABLED)

        # New configuration parameters
        self.new_config_frame = tk.Frame(self.root)
        self.new_config_frame.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_width = tk.Label(self.new_config_frame, text="Width:", font=("Helvetica", self.font_size))
        self.lbl_width.pack(side=tk.LEFT, padx=5, pady=5)
        self.width = tk.IntVar(value=192)
        self.width_entry = tk.Entry(self.new_config_frame, textvariable=self.width, font=("Helvetica", self.font_size))
        self.width_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_height = tk.Label(self.new_config_frame, text="Height:", font=("Helvetica", self.font_size))
        self.lbl_height.pack(side=tk.LEFT, padx=5, pady=5)
        self.height = tk.IntVar(value=192)
        self.height_entry = tk.Entry(self.new_config_frame, textvariable=self.height, font=("Helvetica", self.font_size))
        self.height_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_overlap_width_ratio = tk.Label(self.new_config_frame, text="Overlap Width Ratio:",
                                                font=("Helvetica", self.font_size))
        self.lbl_overlap_width_ratio.pack(side=tk.LEFT, padx=5, pady=5)
        self.overlap_width_ratio = tk.DoubleVar(value=0.2)
        self.overlap_width_ratio_entry = tk.Entry(self.new_config_frame, textvariable=self.overlap_width_ratio,
                                                  font=("Helvetica", self.font_size))
        self.overlap_width_ratio_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_overlap_height_ratio = tk.Label(self.new_config_frame, text="Overlap Height Ratio:",
                                                 font=("Helvetica", self.font_size))
        self.lbl_overlap_height_ratio.pack(side=tk.LEFT, padx=5, pady=5)
        self.overlap_height_ratio = tk.DoubleVar(value=0.2)
        self.overlap_height_ratio_entry = tk.Entry(self.new_config_frame, textvariable=self.overlap_height_ratio,
                                                   font=("Helvetica", self.font_size))
        self.overlap_height_ratio_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_base_detector = tk.Label(self.new_config_frame, text="Base Detector:", font=("Helvetica", self.font_size))
        self.lbl_base_detector.pack(side=tk.LEFT, padx=5, pady=5)
        self.base_detector = tk.StringVar(value=self.detector_mode_options[0])
        self.base_detector_dropdown = tk.OptionMenu(self.new_config_frame, self.base_detector, *self.detector_mode_options)
        self.base_detector_dropdown.config(font=("Helvetica", self.font_size))
        self.base_detector_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

        # IoU and mAP display
        self.metrics_frame = tk.Frame(self.root)
        self.metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        # Use grid geometry manager for better control over layout
        self.metrics_frame.columnconfigure(0, weight=0)
        self.metrics_frame.columnconfigure(1, weight=1)

        self.lbl_iou = tk.Label(self.metrics_frame, text="IoU Scores:", font=("Helvetica", self.font_size))
        self.lbl_iou.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.iou_text = tk.Text(self.metrics_frame, height=2, font=("Helvetica", self.font_size))
        self.iou_text.grid(row=0, column=1, padx=10, pady=10, sticky='we')
        self.iou_text.config(state=tk.DISABLED)
       
        self.lbl_map = tk.Label(self.metrics_frame, text="mAP Score:", font=("Helvetica", self.font_size))
        self.lbl_map.grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.map_text = tk.Text(self.metrics_frame, height=1, font=("Helvetica", self.font_size))
        self.map_text.grid(row=1, column=1, padx=10, pady=10, sticky='we')
        self.map_text.config(state=tk.DISABLED)

    def update_font_size(self, font_size):
        """
        Update the font size of the UI elements.

        Args:
            font_size (int): New font size.
        """
        self.font_size = font_size
        self.menu_bar.config(font=("Helvetica", self.font_size))
        for menu in self.menu_bar.winfo_children():
            menu.config(font=("Helvetica", self.font_size))
        self.btn_prev_image.config(font=("Helvetica", self.font_size))
        self.btn_next_image.config(font=("Helvetica", self.font_size))
        self.btn_segment.config(font=("Helvetica", self.font_size))
        self.lbl_segmentation_mode.config(font=("Helvetica", self.font_size))
        self.segmentation_mode_dropdown.config(font=("Helvetica", self.font_size))
        self.lbl_score_threshold.config(font=("Helvetica", self.font_size))
        self.score_threshold_spinbox.config(font=("Helvetica", self.font_size))
        self.lbl_current_config.config(font=("Helvetica", self.font_size))
        self.meta_info_text.config(font=("Helvetica", self.font_size))
        self.lbl_width.config(font=("Helvetica", self.font_size))
        self.width_entry.config(font=("Helvetica", self.font_size))
        self.lbl_height.config(font=("Helvetica", self.font_size))
        self.height_entry.config(font=("Helvetica", self.font_size))
        self.lbl_overlap_width_ratio.config(font=("Helvetica", self.font_size))
        self.overlap_width_ratio_entry.config(font=("Helvetica", self.font_size))
        self.lbl_overlap_height_ratio.config(font=("Helvetica", self.font_size))
        self.overlap_height_ratio_entry.config(font=("Helvetica", self.font_size))
        self.lbl_base_detector.config(font=("Helvetica", self.font_size))
        self.base_detector_dropdown.config(font=("Helvetica", self.font_size))
        self.lbl_iou.config(font=("Helvetica", self.font_size))
        self.iou_text.config(font=("Helvetica", self.font_size))
        self.lbl_map.config(font=("Helvetica", self.font_size))
        self.map_text.config(font=("Helvetica", self.font_size))

    def display_image(self, image, canvas='left'):
        """
        Display an image on the specified canvas.

        Args:
            image (Image): Image to be displayed.
            canvas (str): Canvas to display the image ('left' or 'right').
        """
        self.delete_previous_image(canvas)
        if canvas == 'left':
            self.tk_image = ImageTk.PhotoImage(image)
            self.left_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.left_canvas.config(scrollregion=self.left_canvas.bbox(tk.ALL))
            self.images_to_delete.append(self.tk_image)
        elif canvas == 'right':
            self.tk_gt_image = ImageTk.PhotoImage(image)
            self.right_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_gt_image)
            self.right_canvas.config(scrollregion=self.right_canvas.bbox(tk.ALL))
            self.images_to_delete.append(self.tk_gt_image)

    def delete_previous_image(self, canvas):
        """
        Delete the previous image from the specified canvas to free memory.

        Args:
            canvas (str): Canvas to delete the image from ('left' or 'right').
        """
        if canvas == 'left':
            if hasattr(self, 'tk_image') and self.tk_image is not None:
                self.root.after(0, self.tk_image.__del__)
        elif canvas == 'right':
            if hasattr(self, 'tk_gt_image') and self.tk_gt_image is not None:
                self.root.after(0, self.tk_gt_image.__del__)

    def update_meta_info(self, info):
        """
        Update the metadata information displayed.

        Args:
            info (str): Metadata information to be displayed.
        """
        self.meta_info_text.config(state=tk.NORMAL)
        self.meta_info_text.delete(1.0, tk.END)
        self.meta_info_text.insert(tk.END, info)
        self.meta_info_text.config(state=tk.DISABLED)

    def get_zoom_level(self, canvas):
        """
        Get the current zoom level for the specified canvas.

        Args:
            canvas (str): Canvas to get the zoom level for ('left' or 'right').

        Returns:
            float: Current zoom level.
        """
        return self.zoom_levels[canvas]

    def zoom_in(self, canvas):
        """
        Zoom in on the specified canvas.

        Args:
            canvas (str): Canvas to zoom in on ('left' or 'right').
        """
        self.zoom_levels[canvas] *= 1.1

    def zoom_out(self, canvas):
        """
        Zoom out on the specified canvas.

        Args:
            canvas (str): Canvas to zoom out on ('left' or 'right').
        """
        self.zoom_levels[canvas] /= 1.1

    def reset_zoom_level(self, canvas):
        """
        Reset the zoom level for the specified canvas to the default value.

        Args:
            canvas (str): Canvas to reset the zoom level for ('left' or 'right').
        """
        self.zoom_levels[canvas] = 1.0

    def display_ious(self, ious):
        """
        Display IoU scores.

        Args:
            ious (list): List of IoU scores.
        """
        self.iou_text.config(state=tk.NORMAL)
        self.iou_text.delete(1.0, tk.END)
        self.iou_text.insert(tk.END, ', '.join([f'{iou:.2f}' for iou in ious]))
        self.iou_text.config(state=tk.DISABLED)

    def display_map(self, map_score):
        """
        Display the mean Average Precision (mAP) score.

        Args:
            map_score (float): mAP score to be displayed.
        """
        self.map_text.config(state=tk.NORMAL)
        self.map_text.delete(1.0, tk.END)
        if map_score is not None:
            self.map_text.insert(tk.END, f'{map_score:.2f}')
        self.map_text.config(state=tk.DISABLED)
