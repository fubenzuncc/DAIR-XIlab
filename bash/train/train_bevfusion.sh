# python v2x/train_bevfusion_base.py \
#     configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml\
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
#     --load_from pretrained/lidar-only-det.pth 

WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}
INFRA_CONFIG_FILE=./configs/vic3d/late-fusion-image-pointcloud/bevfusion/trainval_config_i_v1.py

python v2x/train_bev.py \
     ${INFRA_CONFIG_FILE} 