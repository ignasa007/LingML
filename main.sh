for dataset in {aaai-constraint-covid-appended,aaai-constraint-covid}; do
    python3 -B main.py \
        --dataset ${dataset} \
        --model xlm-mlm-en-2048 \
        ADD_NEW_TOKENS False \
        DATA.SAVE_EVERY None \
        DEVICE_INDEX 1;
done