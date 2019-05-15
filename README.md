
# Reproducibility Challenge
- Chosen paper: [FAST AND ACCURATE TEXT CLASSIFICATION: SKIMMING, REREADING AND EARLY STOPPING](https://openreview.net/forum?id=ryZ8sz-Ab)
- Referenced paper: [LEARN TO SKIM](https://arxiv.org/abs/1704.06877)

## Introduction
We implemented two baseines and proposed model from the chosen paper and two variances which used the reward definition from referenced paper as comparison. IMDB dataset was used.

Models in the paper:
- whole reading model
- early stopping model
- skim, reread and early stopping

Models as comparison:
- early stopping model with different reward
- skim, reread and early stopping with different reward
## Requirement
see file [requirements.txt](requirements.txt)
## Usage
`python [model_name.py] [--parameter value]`

```
--seed  # random seed
--alpha # trade off between efficiency(computation cost measured in FLOPs) and accuracy
--gamma # discount factor
```

Example: 
```
python whole_reading.py
python whole_reading.py --seed 2019
python skim_reread_es_main.py --alpha 0.2 --gamma 0.95
```
## Notes
We are still testing the [whole_reading.py](whole_reading.py), currently it tends to breakdown with the error:"out of memory". Therefore we have provided a [jupyter_notebook](../master/jupyter_notebook) version which runs correctly under Google Colab with GPU turned on.


