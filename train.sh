set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/gaowenyang/DELLA:$PYTHONPATH
cd /gaowenyang/DELLA
#$ThisDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

##Conditional Generation
# python -u main.py \
#     --train_file ./data/yelp/yelp.train.txt \
#     --valid_file ./data/yelp/yelp.valid.txt \
#     --dataset_type wp \
#     --per_gpu_train_batch_size 64 \
#     --pretrained_model ../models/gpt2 \
#     --model_name della \
#     --cycle_annealing \
#    --n_gpus 2 \

#Unconditional Generation
python -u main.py \
    --train_file ./data/yelp/yelp.train.txt \
    --valid_file ./data/yelp/yelp.valid.txt \
    --per_gpu_train_batch_size 16 \
    --model_name della \
    --pretrained_model ../models/gpt2 \
    --cycle_annealing \
    |& tee log.txt
