conda create -n c5 python=3.10 -y
conda activate c5
pip install torch torchvision
pip install transformers
pip install ultralytics
pip install albumentations
pip install pycocotools
pip install opencv-python
echo "Done"
