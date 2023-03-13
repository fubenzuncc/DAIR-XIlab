WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}


# mkdir ./data/DAIR-V2X
# DATA_ORIGIN_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure
# ln -s DATA_ORIGIN_DIR ${WORKSPACE}/data/DAIR-V2X

#bash ./bash/eval/eval_lidar_late_fusion_pointpillars.sh 0 late_fusion 0 0 100 --no-comp
bash ./bash/eval/eval_single.sh 0 veh_only 0 0 100 --no-comp

# bash ./bash/eval/eval_lidar_late_fusion_pointpillars_testA.sh 0 late_fusion 0 0 100 --no-comp

# 
# 
# car 3d IoU threshold 0.30, Average Precision = 65.55
# car 3d IoU threshold 0.50, Average Precision = 55.98
# car 3d IoU threshold 0.70, Average Precision = 40.01
# car bev IoU threshold 0.30, Average Precision = 67.39
# car bev IoU threshold 0.50, Average Precision = 62.00
# car bev IoU threshold 0.70, Average Precision = 54.11
# Average Communication Cost = 478.56 Bytes