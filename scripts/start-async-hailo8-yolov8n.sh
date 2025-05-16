#!/bin/bash

python3 main.py \
-c=10 \
-t=10 \
-spath=sdk/samples/videos \
-m=sdk/models/object-detection/yolo/hailo-8-hef/yolov8n.hef \
-f \
-l \
-d \
--hailo-async
