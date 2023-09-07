for dataset in {aaai-constraint-covid-appended,aaai-constraint-covid}; do
    for run in {1..5}; do
        for ADD_NEW_TOKENS in {False,True,}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model longformer-base-4096 \
                ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX 1;
        done
    done
done