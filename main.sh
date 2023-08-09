for model in {covid-twitter-bert-v2,twitter-roberta-base-sentiment-latest}; do
    for ADD_NEW_TOKENS in {True,False}; do
        for i in {0..5}; do
            python3 -B main.py \
                --model ${model} \
                --dataset aaai-constraint-covid-filtered \
                ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done