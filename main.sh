for dataset in {aaai-constraint-covid-appended,aaai-constraint-covid}; do
    for run in {1..5}; do
        for model in {xlm-roberta-base,}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model ${model} \
                ADD_NEW_TOKENS False \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done

for dataset in {aaai-constraint-covid-appended,aaai-constraint-covid}; do
    for run in {1..5}; do
        for model in {xlm-roberta-base,xlm-mlm-en-2048,}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model ${model} \
                ADD_NEW_TOKENS True \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done