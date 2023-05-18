import os
import cv2
import time
import torch
import random
import argparse
from glob import glob

from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn
import numpy as np

import ffmpeg

# classes and randomly generated colors
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colors = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

# default arguments
parser = argparse.ArgumentParser(description="Mask-RCNN (segmentation model) implementation in PyTorch")
output_group = parser.add_mutually_exclusive_group()
boxes_group = parser.add_mutually_exclusive_group()
masks_group = parser.add_mutually_exclusive_group()
labels_group = parser.add_mutually_exclusive_group()
parser.add_argument("--grey-background","-g",action="store_true",help="make the background monochromatic")
parser.add_argument("--classes","-c",nargs="+",default=["all"],help="limit to certain classes (all or see classes.txt)")
parser.add_argument("--detection-threshold",default=0.7,type=float,help="confidence threshold for detection (0-1)")
parser.add_argument("--mask-threshold",default=0.5,type=float,help="confidence threshold for segmentation mask (0-1)")
parser.add_argument("--max-detections",default=0,type=int,help="maximum concurrent detections (leave 0 for unlimited)")
output_group.add_argument("--hide-output",action="store_true",help="do not show output")
output_group.add_argument("--display-title",default="Mask-RCNN",help="window title")
boxes_group.add_argument("--hide-boxes",action="store_true",help="do not show bounding boxes")
masks_group.add_argument("--hide-masks",action="store_true",help="do not show segmentation masks")
labels_group.add_argument("--hide-labels",action="store_true",help="do not show labels")
masks_group.add_argument("--mask-opacity",default=0.4,type=float,help="opacity of segmentation masks")
parser.add_argument("--show-fps",action="store_true",help="display processing speed (fps)")
labels_group.add_argument("--text-thickness",default=2,type=int,help="thickness of label text")
boxes_group.add_argument("--box-thickness",default=3,type=int,help="thickness of boxes")

# subparsers for different inputs
subparsers = parser.add_subparsers()

folder = subparsers.add_parser("folder")
output_group = folder.add_mutually_exclusive_group()
folder.add_argument("--input-video","--input","-i",default=".",required=True)
folder.add_argument("--input-folder","--input","-i",default=".",required=True,help="input folder")
output_group.add_argument("--output-folder","--output","-o",default="output/",help="output save location")
output_group.add_argument("--no-save",action="store_true",help="do not save output images")
folder.add_argument("--extensions","-e",nargs="+",default=["png", "jpeg", "jpg", "bmp", "tiff", "tif"],help="image file extensions")
folder.set_defaults(action="folder")

# parse args
args = parser.parse_args()
#include_classes = classes[1:] if "all" in args.classes else args.classes
include_classes = ["BG", "chair"]
mode = args.action

# load model
model = maskrcnn(pretrained=True).eval()

def detect(image):
    # feed forward the image
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    cover = np.zeros(image.shape,dtype=bool)
    i = 0
    for box, label, score, mask in zip(*output.values()):
        # check if we need to keep detecting
        if score < args.detection_threshold or (i >= args.max_detections and args.max_detections != 0):
            break
        # ignore irrelevant classes
        if not classes[label] in include_classes:
            continue
        i += 1
        # draw mask
        image[mask[0] > args.mask_threshold] = image[mask[0] > args.mask_threshold] * (1 - args.mask_opacity) + args.mask_opacity * np.array(colors[label])
        # update the cover
        cover[mask[0] > args.mask_threshold] = 1
    # make the background grey
    ret, thresh = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY)
    image[~cover] = np.tile(np.expand_dims(thresh,axis=2),(1,1,3))[~cover]
    image[thresh == 255] = 0
    return image

def output_video():
    frame_dir = args.output_folder
    output_file = 'output.mp4'
    if os.path.exists(output_file):
        os.remove(output_file)
    input_stream = ffmpeg.input(f'{frame_dir}/frame%d.jpg')
    output_stream = ffmpeg.output(input_stream, output_file)
    ffmpeg.run(output_stream)

source = args.input_video
# create a video capture
cap = cv2.VideoCapture(source)
if cap is None or not cap.isOpened():
    raise RuntimeError(f"video (\"{source}\") is not a valid input")
if not os.path.exists('video_frames'):
    os.makedirs('video_frames')
skip_frames = 10 # change to skip frames
frame_count = 0
while True:
    success, image = cap.read()
    if not success:
        break
    if frame_count % skip_frames == 0:
        cv2.imwrite(os.path.join('video_frames', f'frame{int(frame_count / skip_frames)}.jpg'), image)
    frame_count += 1
cap.release()

files = []
folder = args.input_folder
# add a "/" to the end of the folder
if not folder.endswith("/") and not folder.endswith("\\"):
    folder += "/"
# create a list of files
for extension in args.extensions:
    files += glob(f"{folder}**/*.{extension}",recursive=True)
i = 0

while True:
    # get the path and image
    path = files[i]
    image = cv2.imread(path)
    i += 1 # increment counter
    
    # run the detection
    image = detect(image)
    # show the image
    if not args.hide_output:
        cv2.imshow(args.display_title,image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # save output
    if not args.no_save:
        save_path = os.path.join(args.output_folder,os.path.relpath(path,folder))
        # save image
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory,exist_ok=True)
        cv2.imwrite(save_path,image)
        if i >= len(files):
            break

output_video()
