for model in {bertweet-covid19-base-uncased,longformer-base-4096,twitter-roberta-base-sentiment-latest,xlnet-base-cased,bert-base-uncased,xlm-roberta-base,roberta-base,distilbert-base-uncased,albert-base-v2,xlm-mlm-en-2048,covid-twitter-bert-v2}; do
    for dataset in {aaai-constraint-covid-appended,aaai-constraint-covid}; do
        for run in {1..5}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model ${model} \
                ADD_NEW_TOKENS False \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done