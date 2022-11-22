MODE=$1
NGPUS=$2
CFG=$3
CKPT_DIR=$4
DATAPATH=$5
PT_OUTPUT_DIR=$CKPT_DIR

if [[ "$MODE" = "klite" ]]; then
    USE_KNOWLEDGE="--use_knowledge"
else
    USE_KNOWLEDGE=""
fi
python -m torch.distributed.launch --nproc_per_node $NGPUS --master_port 12345 main.py --eval \
--cfg $CFG --resume ${CKPT_DIR}/model_state_dict.pt --data-path $DATAPATH $USE_KNOWLEDGE