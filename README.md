# Mask R-CNN PyTorch Implementation
This is a PyTorch implementation of Mask R-CNN that was adapted from https://github.com/George-Ogden/Mask-RCNN. This implementation accepts a video input and outputs the video after performing segmentation on each frame.

## Setup
### pip
`pip install -r requirements.txt`
### conda
`conda env create -f env.yaml`
## Usage
```
usage: main.py [-h] [--grey-background] [--classes CLASSES [CLASSES ...]]
               [--detection-threshold DETECTION_THRESHOLD] [--mask-threshold MASK_THRESHOLD]   
               [--max-detections MAX_DETECTIONS]
               [--hide-output | --display-title DISPLAY_TITLE] [--hide-boxes] [--hide-masks]   
               [--hide-labels] [--mask-opacity MASK_OPACITY] [--show-fps]
               [--text-thickness TEXT_THICKNESS] [--box-thickness BOX_THICKNESS]
               {image,folder,video,webcam} ...

Mask-RCNN (segmentation model) implementation in PyTorch

positional arguments:
  {folder}

optional arguments:
  -h, --help            show this help message and exit
  --grey-background, -g
                        make the background monochromatic
  --classes CLASSES [CLASSES ...], -c CLASSES [CLASSES ...]
                        limit to certain classes (all or see classes.txt)
  --detection-threshold DETECTION_THRESHOLD
                        confidence threshold for detection (0-1)
  --mask-threshold MASK_THRESHOLD
                        confidence threshold for segmentation mask (0-1)
  --max-detections MAX_DETECTIONS
                        maximum concurrent detections (leave 0 for unlimited)
  --hide-output         do not show output
  --display-title DISPLAY_TITLE
                        window title
  --hide-boxes          do not show bounding boxes
  --hide-masks          do not show segmentation masks
  --hide-labels         do not show labels
  --mask-opacity MASK_OPACITY
                        opacity of segmentation masks
  --show-fps            display processing speed (fps)
  --text-thickness TEXT_THICKNESS
                        thickness of label text
  --box-thickness BOX_THICKNESS
                        thickness of boxes
```
### Folder
```
usage: main.py folder [-h] --input-video INPUT_VIDEO --input-folder INPUT_FOLDER [--output-folder OUTPUT_FOLDER | --no-save] [--extensions EXTENSIONS [EXTENSIONS ...]]

arguments:
  -h, --help            show this help message and exit
  --input-video INPUT_VIDEO, --input INPUT_VIDEO, -i INPUT_VIDEO
                        input video
  --input-folder INPUT_FOLDER, --input INPUT_FOLDER, -i INPUT_FOLDER
                        input folder containing video frames
  --output-folder OUTPUT_FOLDER, --output OUTPUT_FOLDER, -o OUTPUT_FOLDER
                        output save location
  --no-save             do not save output video
  --extensions EXTENSIONS [EXTENSIONS ...], -e EXTENSIONS [EXTENSIONS ...]
                        image file extensions
```
## Classes
For a list of classes, see [classes.txt](classes.txt).
