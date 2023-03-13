WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
DATA_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C

cd ${WORKSPACE}

DATA=${DATA_DIR}/cooperative-vehicle-infrastructure
# OUTPUT_DIR=${DATA_DIR}/cache/vic-late-lidar
OUTPUT_DIR=/workspace/mnt/storage/guangcongzheng/zju_fbz_backup/DAIR-V2X-C-test_A/V2X_C_val_cache/vic-late-lidar

rm -r ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/result
mkdir -p ${OUTPUT_DIR}/inf/lidar
mkdir -p ${OUTPUT_DIR}/veh/lidar

INFRA_CONFIG_PATH=./configs/vic3d/late-fusion-pointcloud/second/trainval_config_i.py
INFRA_MODEL_PATH=./checkpoints/infrastructure-side/second_80epochs/latest.pth

VEHICLE_CONFIG_PATH=./configs/vic3d/late-fusion-pointcloud/second/trainval_config_v.py
VEHICLE_MODEL_PATH=./checkpoints/vehicle-side/second_80epochs/trainval_config_v.py

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
  --input $DATA \
  --output $OUTPUT_DIR \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_CONFIG_PATH \
  --inf-model-path $INFRA_MODEL_PATH \
  --veh-config-path $VEHICLE_CONFIG_PATH \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION