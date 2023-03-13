/opt/conda/bin/conda init
source ~/.bashrc
/opt/conda/bin/conda activate


source /opt/conda/etc/porfile.d/conda.sh
conda activate

WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X
cd ${WORKSPACE}



# prepare data
mkdir ./data/DAIR-V2X
DATA_ORIGIN_DIR=/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure
ln -s DATA_ORIGIN_DIR ${WORKSPACE}/data/DAIR-V2X



