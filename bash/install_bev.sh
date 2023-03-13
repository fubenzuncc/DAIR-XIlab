WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}

# 永久换源，一会再恢复
# 清华
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里源
#pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

#pip install spconv-cu114
# pypcd
pip install pypcd3

# mmdet

pip install tqdm
pip install Pillow==8.4.0
pip install torchpack
pip install nuscenes-devkit
#pip install mpi4py==3.0.3
# pip install numba==0.53.0 --use-feature=2020-resolver

#conda install -c conda-forge mpi4py openmpi
conda install -c conda-forge  -y mpi4py==3.0.3 openmpi==4.0.4

pip install mmcv-full==1.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmdet==2.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmsegmentation==0.29.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# pip install mmsegmentation==0.14.1
# pip install mmpycocotools==12.0.3

# 装在DAIR-V2X外面一个目录
# git clone https://kgithub.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
python mmdet3d/utils/collect_env.py

# pip install numba==0.53.0 --use-feature=2020-resolver
# 注释
pip install --upgrade numba --use-feature=2020-resolver -i https://pypi.tuna.tsinghua.edu.cn/simple
# 换回默认源
pip config unset global.index-url

