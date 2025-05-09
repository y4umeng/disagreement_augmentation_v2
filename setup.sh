python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 setup.py develop
pip install -e .

mkdir download_ckpts
cd download_ckpts
wget https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar
tar -xvf cifar_teachers.tar
