WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
DATA_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure
OUTPUT_DIR=/workspace/mnt/storage/guangcongzheng/zju_fbz_backup/DAIR-V2X-val/eval_lidar_image_late_fusion_inf_official_pointpillars_veh_mvxnt_bs2x8_60epochs

rm -r ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/test
mkdir -p ${OUTPUT_DIR}/result
mkdir -p ${OUTPUT_DIR}/inf/lidar
mkdir -p ${OUTPUT_DIR}/veh/lidar

cd ${WORKSPACE}

#INFRA_MODEL_PATH=./checkpoints/infrastructure-side/mvxnt_bs2x8_60epochs
#INFRA_CONFIG_NAME="trainval_config_i.py"
#INFRA_MODEL_NAME="epoch_60.pth"

#INFRA_MODEL_PATH=./checkpoints/infrastructure-side/pointpillars
#INFRA_CONFIG_NAME="trainval_config_i.py"
#INFRA_MODEL_NAME="epoch_80.pth"

INFRA_MODEL_PATH=configs/vic3d/late-fusion-pointcloud/pointpillars
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"

VEHICLE_MODEL_PATH=./checkpoints/vehicle-side/mvxnt_bs2x8_60epochs
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="epoch_60.pth"

#VEHICLE_MODEL_PATH=configs/vic3d/late-fusion-pointcloud/pointpillars
#VEHICLE_CONFIG_NAME="trainval_config_v.py"
#VEHICLE_MODEL_NAME="vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

#VEHICLE_MODEL_PATH=./checkpoints/vehicle-side/pointpillars_80epochs
#VEHICLE_CONFIG_NAME="trainval_config_v.py"
#VEHICLE_MODEL_NAME="epoch_80.pth"

SPLIT_DATA_PATH=./data/split_datas/cooperative-split-data.json
#0 late_fusion 0 0 100 --no-comp
# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=$1
FUSION_METHOD=$2
DELAY_K=$3
EXTEND_RANGE_START=$4
EXTEND_RANGE_END=$5
TIME_COMPENSATION=$6

python v2x/eval.py \
  --input ${DATA_DIR} \
  --output ${OUTPUT_DIR} \
  --model ${FUSION_METHOD} \
  --dataset vic-async \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/${VEHICLE_CONFIG_NAME} \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar_camera \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION