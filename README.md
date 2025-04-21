# Hailo SDK on Ubuntu 22.04

 - models
 includes infrence model from Hailo Model Zoo

 - packages
 includes Hailo runtime required dependency

# install require

install driver and hailort (include hailortcli, hailo-accelerator-integration-tool)
```sh
sudo dkpg -i packages/*.deb
```

install python3.11 for hailo sdk
```sh
sudo apt install -y python3.11 python3.11-dev python3.11-venv
```
init python environment for hailo
```sh
python3 -m venv --system-site-package venv_hailo
source venv_hailo/bin/activate

pip install hailort-4.20.0-cp311-cp311-linux_x86_64.whl
```

for more different type of model file like onnx, tflite
```sh
pip install onnxruntime tflite-runtime
pip install psutil # for monitor cpu usages
```