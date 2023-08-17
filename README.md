# VCN-Fake-News

This is the code repository for the technical part of Project FIDES. As of 17 August, 2023, the repository contains the code for Studies 3 and 4.

## Directory Structure

- <b>assets</b> - plots generated for different experiments. <br>
\<dataset> -> \<model-1>_\<model-2> -> training/validation -> accuracy/f1-score/loss <br>
Note: these plots were made before we decided to average out results over multiple runs.
- <b>config</b> - default configuration files.
- <b>daa_classes</b> - Python classes to handle different datasets, and make them suitable for training.
- <b>datasets</b> - raw datasets in csv format.
- <b>model_classes</b> - Python classes to handle different models.
    - number of labels need to be changed from 3 to 2 for RoEBRTA.
    - both models need special implementation for accommodating language features.
- <b>results</b> - results of the different runs. <br>
\<dataset> -> \<model> -> \<run-date> -> logs/training_results/validation_results <br>
Note: CovidMis20 experiment has only 1 run which was conducted before the decision to conduct multiple runs for each experiment was made. 
- <b>utils</b> - utility functions for running the transformer experiments.
- <b>analysis.ipynb</b> - notebook to consoliate results.
- <b>main.py</b> - main file for running the transformer experiments.
- <b>xml.ipynb</b> - notebook running the experiments using simple machine learning algorithms with the language features.

## Setup

```bash
conda create -n fakenews python=3.8
conda activate fakenews
pip install -r requirements.txt
```

## Execution

To run the transformer experiments, execute
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model>
```

where dataset can be one of
- <b>aaai-contraint-covid</b> - the original AAAI-Constraint dataset
- <b>aaai-contraint-covid-appended</b> - the original AAAI-Constraint dataset with language features
- <b>aaai-contraint-covid-cleaned</b> - cleaned version of the AAAI-Constraint dataset
- <b>aaai-contraint-covid-cleaned-appended</b> - cleaned version of the AAAI-Constraint dataset with language features

and model can be one of
- <b>covid-twitter-bert-v2</b> - latest version of COVID-Twitter-BERT model
- <b>twiiter-roberta-base-sentiment-latest</b> - latest version of Twitter-RoBERTa model

You can also override default configurations using the command line. For example,
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model> \
    ADD_NEW_TOKENS True \
    DATASET.BATCH_SIZE 16 \
    DATASET.args.root <dataset-root> \
    MODEL.MAX_LENGTH 200
```

Note: Make sure to set the device index to <i>None</i> if you do not wish to use the GPU. For example,
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model> \
    DEVICE_INDEX None
```