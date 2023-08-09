for model in {twitter-roberta-base-sentiment-latest,}; do
    for ADD_NEW_TOKENS in {False,True,}; do
        for i in {1..5}; do
            python3 -B main.py \
                --model ${model} \
                --dataset aaai-constraint-covid-filtered-appended \
                ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done