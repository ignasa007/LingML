for model in {covid-twitter-bert-v2,}; do
    for ADD_NEW_TOKENS in {False,True,}; do
        for i in {1..5}; do
            python3 -B main.py \
                --model ${model} \
                --dataset aaai-constraint-covid-filtered-appended \
                ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 0;
        done
    done
done