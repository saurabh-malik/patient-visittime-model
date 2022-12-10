# Patient Visit-Time prediction with Tensorflow

_Author: Saurabh Malik_

## Task

Given an patient's Demographics details, medical data and Hospital attributes, classify the patient's visit duration time.

## Dataset

For the patient visit examples, I have put a sample CSV file "patient-visits.csv" in data/visits folder

```
data/
    visits/
        patient-visits.csv
```


## Quickstart

2. **Baseline** Created a `base_model` directory under the `experiments` directory. It countains a file `params.json` which sets the parameters for the baseline model. It looks like

```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```


4. **Language Model (VLM)** VLM is a language model with a pre-trained Small BERT (L-4_H-512_A-8/1) and fine-tuned with a classification head.

3. **Train** your experiment. Simply run

```
python train.py --data_dir data/visits --model_dir experiments/base_model
```
