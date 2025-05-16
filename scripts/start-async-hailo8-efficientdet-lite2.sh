#!/bin/bash

python3 main.py \
-c=10 \
-t=10 \
-spath=sdk/samples/videos \
-m=sdk/models/object-detection/efficientdet/hailo-8-hef/efficientdet_lite2.hef \
-f \
-l \
-d \
--hailo-async
