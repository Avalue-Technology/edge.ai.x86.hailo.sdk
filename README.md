# edge.ai.x86.hailo.sdk
 - Hailo runtime sdk with HailoRT v4.21.0 [Reference](https://hailo.ai/developer-zone/documentation/hailort-v4-21-0/)
 

# Install dependencies

 - install driver and hailort (include hailortcli, hailo-accelerator-integration-tool)
```sh
sudo dkpg -i packages/*.deb
```

 - install python3.11 for hailo sdk
```sh
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-virtualenv
```

 - init python environment for hailo
```sh
virtualenv venv --python=python3.11
source venv/bin/activate
pip install packages/hailort-*.whl
pip install -r requirements.txt
```

# Know issue
HailoRT v4.21.0 has a memory leak issue. When running for a long time, it will cause the oom killer to force the programe to close or cause the os to crash.

The current workround: After running a video, close the programe and reopen it.