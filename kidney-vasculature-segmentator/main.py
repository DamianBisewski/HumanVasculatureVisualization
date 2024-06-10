# main.py

import tkinter as tk
from controller import SegmentationAppController

if __name__ == "__main__":
    root = tk.Tk()
    detectors = [
        ('configs/r0i.py',
         'checkpoints/r0i.pth',
         'r0 - RTMDet with Mask Head CSPNeXT backbone chkpt 0'),
        ('configs/r0i.py',
         'checkpoints/r1i.pth',
         'r1 - RTMDet with Mask Head CSPNeXT backbone chkpt 1'),
        ('configs/s0i.py',
         'checkpoints/s0i.pth',
         's0 - RTMDet with Mask Head SwinTransformer backbone chkpt 0'),
        ('configs/s0i.py',
         'checkpoints/s1i.pth',
         's1 - RTMDet with Mask Head SwinTransformer backbone chkpt 1'),
        ('configs/m0i.py',
         'checkpoints/m0i.pth',
         'm0 - Mask R-CNN chkpt 0'),
        ('configs/m0i.py',
         'checkpoints/m1i.pth',
         'm1 - Mask R-CNN chkpt 1'),
        ('configs/y0i.py',
         'checkpoints/y0i.pth',
         'y0 - YOLOX with Mask Head chkpt 0'),
        ('configs/y0i.py',
         'checkpoints/y1i.pth',
         'y1 - YOLOX with Mask Head chkpt 1'),
        ('configs/sb0i.py',
         'checkpoints/sb0i.pth',
         'sb0 - RTMDet with Mask Head SwinTransformer backbone COCO chkpt 0'),
        ('configs/sb0i.py',
         'checkpoints/sb1i.pth',
         'sb1 - RTMDet with Mask Head SwinTransformer backbone COCO chkpt 1')
    ]
    masks_detector_index = 5 # m1i.pth
    app = SegmentationAppController(root, detectors, masks_detector_index, 'cuda:0')
    root.mainloop()
