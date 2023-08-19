model='albert-base-v2'
DEVICE_INDEX=0

for run in {1..4}; do
    for dataset in {aaai-constraint-covid,aaai-constraint-covid-appended}; do
        for ADD_NEW_TOKENS in {False,True}; do
            python3 -B main.py \
                --dataset ${dataset} \
                --model ${model} \
                ADD_NEW_TOKENS ${ADD_NEW_TOKENS} \
                DATA.SAVE_EVERY None \
                DEVICE_INDEX ${DEVICE_INDEX};
        done
    done
done