# 1. 开启ssh X11 图像回传Forwarding功能
# vim /etc/ssh/sshd_config
# or

SSHD_CONFIG="/etc/ssh/sshd_config"

if grep -q "^X11Forwarding" $SSHD_CONFIG;then
   sed -i '/^X11Forwarding/s/no/yes/' $SSHD_CONFIG
else
   sed -i '$a X11Forwarding yes' $SSHD_CONFIG
fi

if grep -q "^X11DisplayOffset" $SSHD_CONFIG;then
   sed -i '/^X11DisplayOffset/c X11DisplayOffset 0' $SSHD_CONFIG
else
   sed -i '$a X11DisplayOffset 0' $SSHD_CONFIG
fi

if grep -q "^X11UseLocalhost" $SSHD_CONFIG;then
   sed -i '/^X11UseLocalhost/s/yes/no/' $SSHD_CONFIG
else
   sed -i '$a X11UseLocalhost no' $SSHD_CONFIG
fi

# systemctl restart ssh.service
# or
/etc/init.d/ssh restart
/etc/init.d/ssh reload

# 2. install environment
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes



apt-get update
#apt-get install ffmpeg libsm6 libxext6  -y
#apt-get install libgl1 libxrender1 libglu1-mesa libfontconfig

apt install libqt5gui5

pip install pypcd3
pip install vtk
pip install mayavi
pip install PyQt5

pip config unset global.index-url
conda config --remove-key channels
#pip install vtk==8.1.2


conda config --add channels conda-forge
conda install vtk
conda install pyqt=4
conda install mayavi

pip install vtk==9.0.1 mayavi==4.7.3 PyQt5==5.15.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install vtk==8.1.1
glxinfo |  grep -E " version| string| rendering|display"
export QT_X11_NO_MITSHM=1
export QT_DEBUG_PLUGINS=1
export LIBGL_ALWAYS_INDIRECT=0
source ~/.bashrc
ldd /opt/conda/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so | grep "not found"

#


# see doc/visualization.md

# visualize GT label in image with vehicle data
mkdir -p ${DATA_ROOT}/visualizations/visualize_gt_label_in_image_with_vehicle_data
python tools/visualize/vis_label_in_image.py --path ${DAIR_V2X_C_V} --output-file ${DATA_ROOT}/visualizations/visualize_gt_label_in_image_with_vehicle_data

# visualize GT label in image with infrastructure data
mkdir -p ${DATA_ROOT}/visualizations/visualize_gt_label_in_image_with_vehicle_data
python tools/visualize/vis_label_in_image.py --path ${DAIR_V2X_C_I} --output-file ${DATA_ROOT}/visualizations/visualize_gt_label_in_image_with_infrastructure_data

# visualize GT label in image with vehicle data
python tools/visualize/vis_label_in_3d.py --task pcd_label --pcd-path ${pcd_path} --label-path ${label_json_path}


python tools/visualize/vis_label_in_3d.py --task fusion --path ${DATA_ROOT}/cache/vic-late-lidar --id 000871


WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
DATA_ROOT=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C
DAIR_V2X_C=${DATA_ROOT}/cooperative-vehicle-infrastructure
DAIR_V2X_C_I=${DATA_ROOT}/cooperative-vehicle-infrastructure/infrastructure-side
DAIR_V2X_C_V=${DATA_ROOT}/cooperative-vehicle-infrastructure/vehicle-side

cd ${WORKSPACE}
python tools/visualize/vis_label_in_3d.py --task fusion --path /workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cache/vic-late-lidar --id 871





pcd_path=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.pcd
label_json_path=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side/label/virtuallidar/000009.json
python tools/visualize/vis_label_in_3d.py --task pcd_label --pcd-path ${pcd_path} --label-path ${label_json_path}

source ~/.bashrc
conda activate dairv
WORKSPACE=/home/yehao/DAIR-V2X
DATA_DIR=/4T/yehao/DAIR-V2X-C
pcd_path=${DATA_DIR}/cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.pcd
label_json_path=${DATA_DIR}/cooperative-vehicle-infrastructure/infrastructure-side/label/virtuallidar/000009.json
QT_DEBUG_PLUGINS=1 python ${WORKSPACE}/tools/visualize/vis_label_in_3d.py --task pcd_label --pcd-path ${pcd_path} --label-path ${label_json_path}


ldconfig -p | grep libGL.so.1
ldconfig -p | grep -i gl.so
apt install libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6


apt-get update
apt-get install mesa-utils x11-apps

defaults read org.xquartz.X11
defaults write org.xquartz.X11 enable_iglx -bool true
defaults write org.xquartz.X11 enable_iglx -bool false

docker build -t reg.supremind.info/algorithmteam/suprevision/zju_zgc:OpenGL .
docker login reg.supremind.info
UcbkuxRjhcgzM4sv1k9_

export LD_PRELOAD=/usr/lib/i386-linux-gn/libGL.so
export LD_PRELOAD=/usr/lib/i386-linux-gnu/libGL.so.1
export LD_LIBRARY_PATH=/usr/lib/i386-linux-gn

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
echo '/usr/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf
echo '/usr/lib/x86_64-linux-gnu/libGL.so.1' >> /etc/ld.so.preload
echo '/usr/lib/x86_64-linux-gnu/libEGL.so.1' >> /etc/ld.so.preload

find /usr -iname "*libGL.so*" -exec ls -l {} \;

find /usr -iname "*libGL.so*" -exec ls -l {} \;

find /usr -iname "*nvidia-4" -exec ls -l {} \; | grep GL


docker run --rm -it nvidia/opengl:1.2-glvnd-devel-ubuntu20.04

docker pull nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04


ls /usr/local
/usr/src/nvidia-418.87.00
find /usr -ipath *nvidia* -type d
find /usr -path *418.87.00*
find /usr -path *460.*
find /usr -path *460.*
apt install nvidia-driver-440

export MESA_DEBUG=1
export QT_X11_NO_MITSHM=1

LIBGL_DEBUG=1 glxinfo


apt-get update \
  && apt-get install -y -qq glmark2 \
  && glmark2

ldd $(which glxinfo) | grep libGL