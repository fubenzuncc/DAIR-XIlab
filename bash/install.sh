WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}

# 永久换源，一会再恢复
# 清华
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# 阿里源
#pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# pypcd
pip install pypcd3

# mmdet
pip install mmcv-full==1.3.14
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install mmpycocotools==12.0.3

# 装在DAIR-V2X外面一个目录
cd ..
# git clone https://kgithub.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
# git checkout v0.17.1
#pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v -e .
python mmdet3d/utils/collect_env.py

# 注释

# 换回默认源
pip config unset global.index-url

