# Patient Visit-Time prediction with Tensorflow

_Author: Saurabh Malik_

## Task

Given an patient's animal class, breed, reason for appointment,  predict the correct label.

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

For every future experiments, we will create a new directory under `experiments` with a similar `params.json` file. 

3. **Train** your experiment. Simply run

```
python train.py --data_dir data/visits --model_dir experiments/base_model
```
