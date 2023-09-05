for dataset in {aaai-constraint-covid,aaai-constraint-covid-appended}; do
    for run in {1..5}; do
        for model in {roberta-base,twitter-roberta-base-sentiment-latest}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model ${model} \
                ADD_NEW_TOKENS False \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 0;
        done
    done
done