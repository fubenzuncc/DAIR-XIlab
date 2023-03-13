WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}

# mkdir ./data/DAIR-V2X
# DATA_ORIGIN_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure
# ln -s DATA_ORIGIN_DIR ${WORKSPACE}/data/DAIR-V2X

# cd ./configs/vic3d/late-fusion-pointcloud/pointpillars

# INFRA_CONFIG_FILE=./configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_i.py
# VEHICLE_CONFIG_FILE=./configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v.py

INFRA_CONFIG_FILE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/configs/vic3d/late-fusion-image-pointcloud/mvxnet/trainval_config_i.py

# save_path=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/train_model

python v2x/train.py \
  ${INFRA_CONFIG_FILE} \