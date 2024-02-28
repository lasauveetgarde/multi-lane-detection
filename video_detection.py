import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import os
import time

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names


parser = argparse.ArgumentParser()
parser.add_argument(
    '--no-boxes',
    action='store_true',
    help='do not show bounding boxes, only show segmentation map'
)
args = parser.parse_args()

VIDEO_PATH = 'input/inference_data/video_3.mp4'
# VIDEO_PATH = 0

WEIGHTS_PATH = 'outputs/training/road_line/model_15.pth'
threshold = 0.9

OUT_DIR = os.path.join('outputs', 'inference')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    pretrained=False, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

ckpt = torch.load(WEIGHTS_PATH)
model.load_state_dict(ckpt['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        orig_image = image.copy()
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        start_time = time.time()
        masks, boxes, labels = get_outputs(image, model, threshold)
        # print(boxes)
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
        cv2.putText(
            result,
            text=f"{fps:.1f} FPS",
            org=(15, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.imshow('Result', np.array(result))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")