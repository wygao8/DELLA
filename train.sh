export CUDA_VISIBLE_DEVICES=0,1
$ThisDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python -u main.py \
    --train_file $ThisDir/data/yelp/yelp.train.txt \
    --valid_file $ThisDir/data/yelp/yelp.valid.txt \
    --dataset_type wp \
    --per_gpu_train_batch_size 128 \
    --model_name della \
    --cycle_annealing \
    --n_gpus 2 \
