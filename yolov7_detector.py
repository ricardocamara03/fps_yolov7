#Yolo imports
import sys
sys.path.append('./yolo_v7/models')
sys.path.append('./yolo_v7/utils')
sys.path.append('./yolo_v7/aws')
sys.path.append('./yolo_v7/google_app_engine')
sys.path.append('./yolo_v7/wandb_logging')

import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized #TracedModel

class yolov7_detector:
    def __init__(self, model_path = "./models/yolov7-tiny-cone-02.pt"):
        #Begin of yolo initializers
        # Initialize
        set_logging()
        self.device = select_device('')
        # Load model
        self.model  = attempt_load(model_path, self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz  = check_img_size(640, s=self.stride)  # check img_size
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        names  = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w   = old_img_h = imgsz
        old_img_b   = 1
        detections  = []
        #End of yolo initializers

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio     = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh    = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh    = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio     = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return img, ratio, (dw, dh)

    def yolo_detect(self, img):
        # Padded resize
        frame_shape = img.shape
        img = self.letterbox(img, 640, self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.75, 0.60, None, False)
        detections = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_shape).round()
                x1 = int(det[0][0].cpu().numpy())
                y1 = int(det[0][1].cpu().numpy())
                x2 = int(det[0][2].cpu().numpy())
                y2 = int(det[0][3].cpu().numpy())

                detections.append([x1, y1, x2, y2])

        return detections
