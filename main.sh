DEVICE_INDEX=1

for run in {1..1}; do
    for dataset in {covid-misinformation,}; do
        for model in {albert-base-v2,distilbert-base-uncased,}; do
            for ADD_NEW_TOKENS in {True,}; do
                python3 -B main.py \
                    --dataset ${dataset} \
                    --model ${model} \
                    ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                    DATA.SAVE_EVERY None \
                    DEVICE_INDEX ${DEVICE_INDEX};
            done
        done
    done
done