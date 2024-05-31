# Horizon Scout

A collection of scripts and utils relating to the mission classifiers for Horizon Scout project.

## Skypilot

`train_bert.py` and `inference_bert.py` scripts are intended to be used with skypilot.

Here are some useful example skypilot commands:

To launch an instance, name it `g5`, run setup (copy credentials, install poetry) and train the BERT mission classifiers run:
```bash
sky launch -c g5 discovery_utils/horizon_scout/train_bert.yml
```
If `g5` is already launched, use `exec` (this avoids having to run the setup steps) to train the BERT mission classifiers run :
```bash
sky exec g5 discovery_utils/horizon_scout/train_bert.yml
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
