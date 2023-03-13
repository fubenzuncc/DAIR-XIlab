WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}
bash bash/install_bev.sh


CONFIG=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/configs/vic3d/late-fusion-image-pointcloud/mvxnet/trainval_config_i.py


GPUS=8
NNODES=1
NODE_RANK=0
PORT=$((RANDOM + 10000))
MASTER_ADDR=localhost

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    v2x/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

