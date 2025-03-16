#!/bin/bash
# This script is used for training and benchmarking
# the baseline method with SFWSOD on RGBD SOD using
# scribble annotations. Users could also modify from this
# script for their use case.
#
# Usage:
#   # From SFWSOD/ directory.
#   source bashscripts/sod/train_sfnet_rgbd_scribble_sod.sh
#
#

# check use which gpu
if [ $# -ne 1 ];then
  echo "usage: $0 gpu_id"
  exit -1
fi

# Set up parameters for network.
BACKBONE_TYPES=rgbd_sod_sfnet
EMBEDDING_DIM=64

# Set up parameters for training.
PREDICTION_TYPES=segsort
TRAIN_SPLIT=train+
GPUS=0
LR_POLICY=poly
USE_SYNCBN=false
SNAPSHOT_STEP=21850
MAX_ITERATION=109250
WARMUP_ITERATION=1000
LR=1e-4
WD=5e-4
DECAY_RATE=0.1
DECAY_EPOCH=60
BATCH_SIZE=4
CROP_SIZE=224
MEMORY_BANK_SIZE=2
KMEANS_ITERATIONS=10
KMEANS_NUM_CLUSTERS=6
SEM_ANN_LOSS_TYPES=segsort # segsort / none
SEM_OCC_LOSS_TYPES=segsort # segsort / none
IMG_SIM_LOSS_TYPES=segsort # segsort / none
FEAT_AFF_LOSS_TYPES=none # segsort / none
SEM_ANN_CONCENTRATION=6
SEM_OCC_CONCENTRATION=12
IMG_SIM_CONCENTRATION=16
FEAT_AFF_CONCENTRATION=0
SEM_ANN_LOSS_WEIGHT=1.0
SEM_OCC_LOSS_WEIGHT=0.0
IMG_SIM_LOSS_WEIGHT=0.1
FEAT_AFF_LOSS_WEIGHT=0.0
FEAT_CRF_LOSS_WEIGHT=0.0
L1_CRF_LOSS_WEIGHT=0.0
PARTED_CE_LOSS_WEIGHT=1.0
SSC_LOSS_WEIGHT=1.0

# Set up parameters for inference.
INFERENCE_SPLIT=val
INFERENCE_IMAGE_SIZE=224
INFERENCE_CROP_SIZE_H=224
INFERENCE_CROP_SIZE_W=224
INFERENCE_STRIDE=224

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/sod_scribble/${BACKBONE_TYPES}_${PREDICTION_TYPES}/p${CROP_SIZE}_dim${EMBEDDING_DIM}_nc\
${KMEANS_NUM_CLUSTERS}_ki${KMEANS_ITERATIONS}_lr${LR}_syncbn_${USE_SYNCBN}_bs${BATCH_SIZE}_gpu${GPUS}\
_it${MAX_ITERATION}_wd${WD}_memsize${MEMORY_BANK_SIZE}_losstype${SEM_ANN_LOSS_TYPES}_${SEM_OCC_LOSS_TYPES}\
_${IMG_SIM_LOSS_TYPES}_${FEAT_AFF_LOSS_TYPES}_lossconc${SEM_ANN_CONCENTRATION}_${SEM_OCC_CONCENTRATION}\
_${IMG_SIM_CONCENTRATION}_${FEAT_AFF_CONCENTRATION}_lossw${SEM_ANN_LOSS_WEIGHT}_${SEM_OCC_LOSS_WEIGHT}\
_${IMG_SIM_LOSS_WEIGHT}_${FEAT_AFF_LOSS_WEIGHT}_${FEAT_CRF_LOSS_WEIGHT}_${L1_CRF_LOSS_WEIGHT}\
_${PARTED_CE_LOSS_WEIGHT}_${SSC_LOSS_WEIGHT}
echo ${SNAPSHOT_DIR}

# Set up the procedure pipeline.
IS_CONFIG_EMB=1
IS_TRAIN_EMB=1
IS_CONFIG_CLASSIFIER=0
IS_ANNOTATION_1=0
IS_TRAIN_CLASSIFIER_1=0
IS_INFERENCE_CLASSIFIER_1=0
IS_BENCHMARK_CLASSIFIER_1=0

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory and file list.
DATAROOT=dataroot
PRETRAINED=
TEST_DATA_LIST_ROOT=datasets/sod/RGBD

# Train for the embedding.
if [ ${IS_TRAIN_EMB} -eq 1 ]; then
  python pyscripts/benchmark/benchmark_sod_featvis.py\
    --data_dir ${DATAROOT}\
    --test_list_root ${TEST_DATA_LIST_ROOT}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}_featvis\
    --cfg_path ${SNAPSHOT_DIR}/config_sod_emb.yaml\
	--init_gpu_id $1

  python pyscripts/benchmark/benchmark_sod.py\
    --data_dir ${DATAROOT}\
    --test_list_root ${TEST_DATA_LIST_ROOT}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}_softmax\
    --cfg_path ${SNAPSHOT_DIR}/config_sod_emb.yaml\
	--init_gpu_id $1
fi
