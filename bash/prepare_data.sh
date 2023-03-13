

# 创建目录
DATA_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C
cd ${DATA_DIR}
mkdir cooperative-vehicle-infrastructure
mkdir cooperative-vehicle-infrastructure/infrastructure-side
mkdir cooperative-vehicle-infrastructure/vehicle-side
mkdir cooperative-vehicle-infrastructure/cooperative

# 压缩包放在此目录下
# DAIR-V2X-C
# cooperative-vehicle-infrastructure.zip, 245.6MB
# DAIR-V2X-C-I
# cooperative-vehicle-infrastructure-infrastructure-side-image_9823779891974144.zip, 3915.1MB
# cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip, 8874.3MB
# DAIR-V2X-C-V
# cooperative-vehicle-infrastructure-vehicle-side-image_9823779891974144.zip, 2714.7MB
# cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip, 12815MB



# 解压
unzip cooperative-vehicle-infrastructure-vehicle-side-image_9823779891974144.zip \
      -d cooperative-vehicle-infrastructure/vehicle-side
mv cooperative-vehicle-infrastructure/vehicle-side/cooperative-vehicle-infrastructure-vehicle-side-image \
   cooperative-vehicle-infrastructure/vehicle-side/image

unzip cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip \
      -d cooperative-vehicle-infrastructure/vehicle-side
mv cooperative-vehicle-infrastructure/vehicle-side/cooperative-vehicle-infrastructure-vehicle-side-velodyne \
   cooperative-vehicle-infrastructure/vehicle-side/velodyne

unzip cooperative-vehicle-infrastructure-infrastructure-side-image_9823779891974144.zip \
      -d cooperative-vehicle-infrastructure/infrastructure-side
mv cooperative-vehicle-infrastructure/infrastructure-side/cooperative-vehicle-infrastructure-infrastructure-side-image \
   cooperative-vehicle-infrastructure/infrastructure-side/image

unzip cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip \
      -d cooperative-vehicle-infrastructure/infrastructure-side
mv cooperative-vehicle-infrastructure/infrastructure-side/cooperative-vehicle-infrastructure-infrastructure-side-velodyne \
   cooperative-vehicle-infrastructure/infrastructure-side/velodyne

unzip cooperative-vehicle-infrastructure.zip -d ./


# 分割train, val, test
pip install pypcd3 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
DATA_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C
DATA_ORIGIN_DIR=${DATA_DIR}/cooperative-vehicle-infrastructure

python ${WORKSPACE}/tools/dataset_converter/dair2kitti.py \
    --source-root ${DATA_ORIGIN_DIR}/infrastructure-side \
    --target-root ${DATA_ORIGIN_DIR}/infrastructure-side \
    --split-path ${WORKSPACE}/data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view infrastructure --no-classmerge

python ${WORKSPACE}/tools/dataset_converter/dair2kitti.py \
    --source-root ${DATA_ORIGIN_DIR}/vehicle-side \
    --target-root ${DATA_ORIGIN_DIR}/vehicle-side \
    --split-path ${WORKSPACE}/data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view vehicle --no-classmerge



python tools/create_data.py kitti --root-path ~/code/DAIR-V2X-main/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/ --out-dir ~/code/DAIR-V2X-main/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/




