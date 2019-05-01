
# Reproducibility Challenge
- Chosen paper: [FAST AND ACCURATE TEXT CLASSIFICATION: SKIMMING, REREADING AND EARLY STOPPING](https://openreview.net/forum?id=ryZ8sz-Ab)
- Referenced paper: [LEARN TO SKIM](https://arxiv.org/abs/1704.06877)

-------------------------------
## Introduction
We implemented two baseines and proposed model in the chosen paper and two variances adopted the reward definition from referenced paper as comparison. IMDB dataset was used.

Models in the paper:
- whole reading model
- early stopping model
- skim, reread and early stopping

Models as comparison:
- early stopping model with different reward
- skim, reread and early stopping with different reward
## Requirement
see file [requirements.txt](COMP6248-Polaris/blob/master/requirements.txt)
## Useage
`python [model_name.py] {--parameter value |--paramter value}`

Example: 
```
python whole_reading.py --seed 2019
python skim_reread_es_main.py --alpha 0.2 --gamma 0.95
```




