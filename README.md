# LingML

This is the official code repository for the paper **LingML: Linguistic-Informed Machine Learning for Enhanced Fake News Detection**.

## Directory Structure

- **config** - configuration files for different datasets and LLM models
- **data_classes** - Python classes to handle different datasets, and make them suitable for training
- **datasets** - raw datasets in csv format
    - **aaai-constraint-covid** - original dataset by [Patwa et al., 2020](https://arxiv.org/abs/2011.03327/)
    - **aaai-constraint-covid-appended** - original dataset with appended linguistic features retrieved using [LIWC-22](https://www.liwc.app/)
    - **aaai-constraint-covid-cleaned** - dataset constructed by eliminating records identified by [Bee et al., 2023](https://arxiv.org/abs/2310.04237), as being neither true nor false in the context of COVID-19
    - **aaai-constraint-covid-cleaned-appended** - cleaned dataset with appended linguistic features
- **model_classes** - Python classes to handle 11 transformer-based LLMs
    - all models need special implementation for incorporating language features
    - number of output heads needed to be changed from 3 to 2 for Twitter-RoEBRTa
- **results** - results of the different runs <br>
    - **directory structure** - \<dataset> -> \<model> -> \<run-date> -> training logs and <data-split_results>
- **utils** - utility functions for running the transformer experiments
- **analysis.ipynb** - notebook to consolidate results
- **main.py** - main file for running the transformer experiments
- **plotting.ipynb** - notebook to generate plots for results of experiments in **xml.ipynb**
- **xml.ipynb** - notebook running the experiments using simple machine learning algorithms with the language features

## Setup

```bash
conda create --name <env-name> --file requirements.txt python=3.8
conda activate <env-name>
```

## Execution

To run the transformer experiments, execute
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model>
```

where dataset can be one of
- **aaai-constraint-covid**
- **aaai-constraint-covid-appended**
- **aaai-constraint-covid-cleaned**
- **aaai-constraint-covid-cleaned-appended**

and model can be one of 
- **albert-base-v2** - Base version of ALBERT model with a randomly initialized sequence classification head. See [HF model card](https://huggingface.co/albert-base-v2).
- **bart-base** - Base version of BART model. See [HF model card](https://huggingface.co/facebook/bart-base).
- **bert-base-uncased** - Base version of BERT model. See [HF model card](https://huggingface.co/bert-base-uncased).
- **bertweet-covid-19-base-uncased** - Base version of BERTweet model, a RoBERTa model pre-trained on ~850M tweets, ~5M of which were COVID-19 related. See [HF model card](https://huggingface.co/vinai/bertweet-covid19-base-uncased).
- **covid-twitter-bert-v2** - CT-BERT Model, which is a large BERT model pre-trained on ~97M COVID-related tweets. See [HF model card](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2).
- **distilbert-base-uncased** - Base version of DistilBERT model, a distilled version of BERT, i.e. a smaller model trained with BERT as a teacher. See [HF model card](https://huggingface.co/distilbert-base-uncased).
- **longformer-base-4096** - Base version of Longformer model, which is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. See [HF model card](https://huggingface.co/allenai/longformer-base-4096).
- **roberta-base** - Base version of RoBERTa model. See [HF model card](https://huggingface.co/roberta-base).
- **twitter-roberta-base-sentiment-latest** - Base version of Twitter-RoBERTa model, which is a RoBERTa-base model trained on ~124M tweets, and finetuned for sentiment analysis with the TweetEval benchmark. See [HF model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- **xlm-mlm-en-2048** - XLM model trained with masked language modeling (MLM) objective. See [HF model card](https://huggingface.co/xlm-mlm-en-2048).
- **xlm-roberta-base** - Base version of XLM-RoBERTa model, a multilingual version of RoBERTa. See [HF model card](https://huggingface.co/xlm-roberta-base).
- **xlnet-base-uncased** - Base version of XLNet model. See [HF model card](https://huggingface.co/xlnet-base-cased).

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

For inference, execute
```bash
python3 -B inference.py \
    --dataset <dataset> \
    --model <model> \
    --weights <path-to-weights>
```

For example,
```bash
python3 -B inference.py \
    --dataset aaai-constraint-covid \
    --model covid-twitter-bert-v2 \
    --weights "./results/aaai-constraint-covid/CT-BERT/2023-12-19-00-28-08/ckpt5350.pth"
```

Note: Make sure to set the device index to <i>None</i> if you do not wish to use the GPU, i.e.
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model> \
    DEVICE_INDEX None
```

## Citation

If you use this work, kindly cite it as
```
@misc{singh2024lingmllinguisticinformedmachinelearning,
      title={LingML: Linguistic-Informed Machine Learning for Enhanced Fake News Detection}, 
      author={Jasraj Singh and Fang Liu and Hong Xu and Bee Chin Ng and Wei Zhang},
      year={2024},
      eprint={2405.04165},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.04165}, 
}
```
