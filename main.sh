DEVICE_INDEX=0
dataset='covid-misinformation'

# covid-twitter-bert-v2
# twitter-roberta-base-sentiment-latest
# albert-base-v2
# distilbert-base-uncased

for run in {1..1}; do
    for model in {albert-base-v2,covid-twitter-bert-v2}; do
        python3 -B main.py \
            --dataset ${dataset} \
            --model ${model} \
            ADD_NEW_TOKENS True \
            # LR 0.00002 \
            DATA.SAVE_EVERY None \
            DEVICE_INDEX ${DEVICE_INDEX};
    done
done