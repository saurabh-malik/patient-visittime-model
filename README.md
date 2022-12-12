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
1. **Environment Setup** 
    1. Download Conda or MiniConda
    2. Create a conda environment using environment.yml file. 
        ```conda env create -f environment.yml```
    3. Activate conda environment.
        ```conda activate env-visitmodel```

2. **Baseline Model (VDNN) ** VDNN is a dense neural network with a softmax activation in the out layer.

    Created a `base_model` directory under the `experiments` directory. It countains a file `params.json` which sets the parameters for the baseline model.
    It looks like:

```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```

**Train baseline model**
    To train baseline model run the following command:
    ```
    python train.py --model_type DNN --data_dir data/visits data_version=V2

    ```
    **Monitor the training**
    For training monitoring providing support of tensorboard. Currently configured to log training data each epoch. To launch the tensorboard run the following command.
    ```
    tensorboard --logdir experiments/base_model/logs/fit
    ```
3. **Hyper-parameters Search**
    This model supports hyper-parameter fine tuning using grid search model and tensorboard.
    
    Run the following command to start the hyper-parameter tuning process.
    ``` 
    python search_hyperparams.py --model_type DNN --data_dir data/visits --model_dir experiments/base_model
    ```
    
    **Monitor the search**
    Launch the tensorboard for search monitoring.
    ```
    tensorboard --logdir experiments/language_model/logs/fit
    ```

4. **Language Model (VLM)** VLM is a language model with a pre-trained Small BERT (L-4_H-512_A-8/1) and fine-tuned with a classification head.
    Created a `language_model` directory under the `experiments` directory. It countains a file `params.json` which sets the parameters for the VLM model. It looks like

    ```json
    {
        "learning_rate": 3e-5,
        "dropout_rate": 0.1
        "batch_size": 16,
        "num_epochs": 2,
        ...
    }
    ```
    Note: If running under constraint compute/memory, user smaller batch size such as 16 or 32

    Prerequisite:
    Training VLM model assumes availability of Small BERT (L-4_H-512_A-8/1) pretrained on visit texts. If not pretrained it yet. STOP. Go to Pretrain BERT section and resume once pretraining is done.

    **Train VLM model**
    To train baseline model run the following command:
    ```
    python train.py --model_type=VLM --data_dir=data/visits data_version=V2

    ```
    **Monitor the training**
    Launch the tensorboard for training monitoring.
    ```
    tensorboard --logdir experiments/language_model/logs/fit
    ```

3. **Pre-train BERT on Visit Text Data**
    For the Pre-training of the Small BERT (L-4_H-512_A-8/1), first of all we need to generate data from patient visits.
    This data is availabe in pre-train-data.txt file. 
    Note: To create pre-train-data.txt file run the following command.
    ```
    python  dataprocessing_pretrain.py --data_dir data/visits --data_version V2
    ```
    
    
    Run the following command for data generation.

    **Data Generation**
    ```
        python create_pretraining_data.py \
      --input_file=./data/visits/pretrain/pre-train-data.txt \
      --output_file=./trained-models/pretraining_output/tf_examples.tfrecord \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --do_lower_case=True \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5

    ```
    
    Next is the pretraining step.
    **Pre-training**
    ```
    python run_pretraining.py \
      --input_file=./trained-models/pretraining_output/tf_examples.tfrecord  \
      --output_dir=./trained-models/pretraining_output/checkpoint \
      --do_train=True \
      --do_eval=True \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --train_batch_size=32 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=20 \
      --num_warmup_steps=10 \
      --learning_rate=2e-5
    ```
  Note:
 1. _$BERT_BASE_DIR = ../uncased_L-4_H-512_A-8 is the folder where we have downloaded Small BERT (L-4_H-512_A-8/1) for pre-train._
 2. _Files related to BERT pre-pretraining module such as **create_pretraining_data.py**, **run_pretraining.py** etc are the modified version of files from original codebase (https://github.com/google-research/bert) to work with **Tensorflow V2, as original repo is in Tensorflow V1**._
