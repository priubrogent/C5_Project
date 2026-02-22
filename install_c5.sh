conda create -n c5 python=3.10 -y
conda activate c5
pip install torch torchvision
pip install transformers datasets accelerate timm peft
pip install ultralytics
pip install albumentations
pip install pycocotools
echo "Done"
