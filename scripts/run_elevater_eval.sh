############## Configuration section begins ##################
# Pretrain Method: [klite, unicl]
method=$1

# Model Config: [clip_swin_tiny, clip_swin_base]
model_cfg=$2

# Mode: [linear_probe, finetune, zeroshot]
mode=zeroshot

# Use FP32 [default: True]
use_fp32=True

# Model checkpoint
model_ckpt=$3

model_suffix=$4

# Dataset: [caltech101]
# dataset=$4
datasets=('flower102' 'resisc45-clip' 'hateful-memes' 'eurosat-clip' 'voc2007classification' 'fer2013' 'gtsrb' 'fgvc-aircraft-2013b' 'cifar10' 'caltech101' 'dtd' 'patchcamelyon' 'rendered-sst2' 'oxford-iiit-pets' 'food101' 'kitti-distance' 'cifar100' 'stanfordcar' 'country211' 'mnist')
# output directory
output_root=./outputs/
output_dir=./outputs/${method}/${model_suffix}/

############ Configurations for hyperparameter tuning begin ############
# set to True to disable the automatic hyperparameter tuning
# and set the learning rate and weight accordingly below
# This option is only effective for linear probe and finetuning.

disable_hyperparameter_tuning=False
learning_rate=0.1
l2_weight_decay=1e-6

############ Configurations for hyperparameter tuning end   ############

############ Configurations for linear_probe/finetune begin ############

# Random seed: [0,1,2]
random_seed=0

# Shots: {5, 20, 50} for few shot, and -1 for full-shot
num_shots=0

# Whether to init the linear head with the text encoder
init_head_with_text_encoder=True

# whether to merge the encoder and the linear head
merge_encoder_and_proj=False

############ Configurations for linear_probe/finetune end   ############

############ Configurations for adding knowledge begin ############
# Please change the knowledge source accordingly.
if [[ "$method" = "klite" ]]; then
    use_wiktionary_definition=True
else
    use_wiktionary_definition=False
fi

use_wordnet_hierachy=False
use_wordnet_definition=False
use_gpt3=False
use_gpt3_count=0

############ Configurations for adding knowledge end   ############

############## Configuration section ends ##################


# Launching the job......

cd vision_benchmark
for dataset in ${datasets[@]};
do
    if [[ "$mode" = "linear_probe" ]]; then
        python commands/linear_probe.py --ds resources/datasets/$dataset.yaml --model resources/model/$model_cfg.yaml --no-tuning $disable_hyperparameter_tuning --lr $learning_rate --l2 $l2_weight_decay MODEL.CLIP_FP32 $use_fp32 DATASET.NUM_SAMPLES_PER_CLASS $num_shots DATASET.ROOT $output_dir/datasets OUTPUT_DIR $output_dir/$model_cfg/log DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.FREEZE_IMAGE_BACKBONE True TRAIN.INIT_HEAD_WITH_TEXT_ENCODER $init_head_with_text_encoder TRAIN.MERGE_ENCODER_AND_HEAD_PROJ $merge_encoder_and_proj KNOWLEDGE.WORDNET.USE_HIERARCHY $use_wordnet_hierachy KNOWLEDGE.WORDNET.USE_DEFINITION $use_wordnet_definition KNOWLEDGE.WIKITIONARY.USE_DEFINITION $use_wiktionary_definition KNOWLEDGE.GPT3.USE_GPT3 $use_gpt3 KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS $use_gpt3_count TEST.MODEL_FILE $model_ckpt
    elif [[ "$mode" = "finetune" ]]; then
        python commands/finetune.py --ds resources/datasets/$dataset.yaml --model resources/model/$model_cfg.yaml --no-tuning $disable_hyperparameter_tuning --lr $learning_rate --l2 $l2_weight_decay MODEL.CLIP_FP32 $use_fp32 DATASET.NUM_SAMPLES_PER_CLASS $num_shots DATASET.ROOT $output_dir/datasets OUTPUT_DIR $output_dir/$model_cfg/log DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER $init_head_with_text_encoder TRAIN.MERGE_ENCODER_AND_HEAD_PROJ $merge_encoder_and_proj KNOWLEDGE.WORDNET.USE_HIERARCHY $use_wordnet_hierachy KNOWLEDGE.WORDNET.USE_DEFINITION $use_wordnet_definition KNOWLEDGE.WIKITIONARY.USE_DEFINITION $use_wiktionary_definition KNOWLEDGE.GPT3.USE_GPT3 $use_gpt3 KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS $use_gpt3_count TEST.MODEL_FILE $model_ckpt

    elif [[ "$mode" = "zeroshot" ]]; then
        python commands/zeroshot.py --ds resources/datasets/$dataset.yaml --model resources/model/$model_cfg.yaml MODEL.CLIP_FP32 $use_fp32 DATASET.ROOT $output_root OUTPUT_DIR $output_dir/$model_cfg/log KNOWLEDGE.WORDNET.USE_HIERARCHY $use_wordnet_hierachy KNOWLEDGE.WORDNET.USE_DEFINITION $use_wordnet_definition KNOWLEDGE.WIKITIONARY.USE_DEFINITION $use_wiktionary_definition KNOWLEDGE.GPT3.USE_GPT3 $use_gpt3 KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS $use_gpt3_count TEST.MODEL_FILE $model_ckpt
    else
        echo Unknown mode! Please check and set mode to one of {linear_probe, finetune, zeroshot}.
        exit -1
    fi;
done