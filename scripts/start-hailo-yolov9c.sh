#!/bin/bash

python3 ./src/main.py \
-c=10 \
-t=10 \
-f \
-d \
-l \
-spath=samples/videos/09-10-0800-day3.mp4 \
-m=models/object-detection/yolo/hef/yolov9c.hef
