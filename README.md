# HuBMAP Vasculature Segmentation Visualization App

This repository contains a visualization application for the first-place solution from the Kaggle competition "HuBMAP - Hacking the Human Vasculature". The competition focused on segmenting human blood vessel structures from microscopy images, with the goal of aiding medical research and development.
To help users visualize the segmentation results, this app provides an intuitive interface for exploring the output of the winning model.
Key features of this app:
    • Segmentation Visualization: Display of microscopy images with overlaid segmentation masks.
    • SAHI Integration: Uses the SAHI (Slicing Aided Hyper Inference) library for efficient, large-scale image inference, enabling robust performance on high-resolution medical images. SAHI goes through the image first dividing it into smaller tiles and performs segmentation on each tile. This is supposed to increase the accuracy of segmentation and, as mainly the glomeruli are larger and blood vessels smaller, it increases the efficiency of detecting blood vessels.
    • Easy-to-Use Interface: A simple, clean user interface to browse and inspect results interactively.
This repository serves as a demonstration tool for researchers, practitioners, and enthusiasts to explore cutting-edge solutions for biomedical image analysis. 
### Usage of the app
To run the application, the user, being in the app directory, needs to go to the kidney-vasculature-segmentator directory and type:
``` python main.py
After the start of the application, the screen above is the screen that appears. To load an image, the user needs to click File → Load Image / Load Directory.










If the latter option is chosen, it will load all the images in the selected directory. Then the user can switch images using the Previous and Next buttons which can be found on the left directly below the loaded image. 

There is also an option to load a JSON file with the ground truth. If there is a segmented image for the actually loaded image, it will appear in the upper right part of the screen with annotations attached to it. 

The application gives 12 options to conduct segmentation of an image. 5 of them are individual models (RTMDet CSPNeXT, RTMDet SWNTR, Mask R-CNN, YOLOX, RTMDet SWNTR COCO), each accessible in two versions, there is also an option to combine all of them in an MMDet ensemble and the last option is SAHI. SAHI can be used with each of the models. 



The IoU scores have values between 0 and 1 and for each annotation they show how much the annotation produced by the segmentation and the annotation in the ground truth intersect. This is a quotient of the intersection of both annotations and union of them.
The mAP score also has values between 0 and 1 and this is the mean average precision of the segmentator.

### Installation of the application

The application itself can be installed via command line using command:
``` git clone https://github.com/DamianBisewski/HumanVasculatureVisualization
The data for the application can be found at:
https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data, to download the data the user needs to click „Download all” which can be found at the right-hand side of the website.
The checkpoints for the application can be found at:https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060.
The checkpoints should be installed in the kidney-vasculature-segmentator/checkpoints directory in the app directory.





