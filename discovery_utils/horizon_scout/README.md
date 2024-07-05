# Horizon Scout

A collection of scripts and utils relating to the mission classifiers for the Horizon Scout project.

## Training data

The raw training data can be found on S3 at `s3://discovery-iss/data/horizon_scout/classifier_training_data_inputs/`. To add more training, update the files here.

To process the data so it can be used for training, run `make_training_data.py`. This will save training data for each mission to `s3://discovery-iss/data/horizon_scout/classifier_training_data_outputs/`.

## BERT training and inference

To train the BERT mission classifiers, run `train_bert.py`. Running this file will save a classifier for each mission to `s3://discovery-iss/models/horizon_scout/`.

To use the BERT mission classifiers for inference, run `inference_bert.py`. Before running check the constants at the top of the file. If `MODE` is set to `"New"`, it will perform inference on only today's new companies. If `MODE` is set to `"Full"`, it will perform inference on the full Crunchbase dataset. Predicted labels are stored in the label store found at `s3://discovery-iss/data/crunchbase/enriched/label_store.csv`.

`train_bert.py` and `inference_bert.py` scripts are intended to be ran using skypilot. To run them via Skypilot, the related YAML files need to be executed. These files can be found at `discovery_utils/horizon_scout/train_bert.yml` and `discovery_utils/horizon_scout/inference_bert.yml`.

## Skypilot

Here are some useful example skypilot commands:

To launch an instance, name it `g5`, run setup (copy credentials, install poetry) and train the BERT mission classifiers run:
```bash
sky launch -c g5 discovery_utils/horizon_scout/train_bert.yml
```
If `g5` is already launched, use `exec` (this avoids having to run the setup steps) to train the BERT mission classifiers run :
```bash
sky exec g5 discovery_utils/horizon_scout/train_bert.yml
```
To launch an instance, name it `g5`, run setup (copy credentials, install poetry) and use the BERT mission classifiers for inference run:
```bash
sky launch -c g5 discovery_utils/horizon_scout/inference_bert.yml
```
To ssh to the instance:
```bash
ssh g5
```
To stop the instance:
```bash
sky stop g5
```
To terminate the instance:
```bash
sky down g5
```
To check which instances are running:
```bash
sky status
```
See the [skypilot documentation](https://skypilot.readthedocs.io/en/latest/docs/index.html) for more information.
