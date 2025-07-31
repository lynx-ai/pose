pip install fastapi "uvicorn[standard]" Pillow controlnet-aux
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
mkdir /usr/local/lib/python3.10/dist-packages/controlnet_aux/dwpose/yolox_config
wget "https://raw.githubusercontent.com/patrickvonplaten/controlnet_aux/84c6ecd5ad8a4ad781911d18e9545a71bc6b5a4c/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py" -O /usr/local/lib/python3.10/dist-packages/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
mkdir /usr/local/lib/python3.10/dist-packages/controlnet_aux/dwpose/dwpose_config
wget "https://raw.githubusercontent.com/patrickvonplaten/controlnet_aux/84c6ecd5ad8a4ad781911d18e9545a71bc6b5a4c/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py" -O /usr/local/lib/python3.10/dist-packages/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
pip uninstall numpy
pip install "numpy<2"