
# Fast-And-Accurate-Text-Classification-Reproduction
- Chosen paper: [FAST AND ACCURATE TEXT CLASSIFICATION: SKIMMING, REREADING AND EARLY STOPPING](https://openreview.net/forum?id=ryZ8sz-Ab)
- Referenced paper: [LEARN TO SKIM](https://arxiv.org/abs/1704.06877)

## Introduction
We implemented two baseine models and the proposed model from the chosen paper and two variation models which use the reward definition from the referenced paper [LEARN TO SKIM](https://arxiv.org/abs/1704.06877) for the use of comparison.

The models were implemented from scratch.

We only tested the models on the IMDB dataset for the task of sentiment analysis.

Models in the paper:
- whole reading model
- early stopping model
- skim, reread and early stopping

Models for comparison:
- early stopping model with different reward
- skim, reread and early stopping with different reward

## Results
We failed to reproduce the results as shown in the original paper due to several reasons such as missing details for the implementation. More discussion can be found in the report.

For the result of IMDB dataset, our highest accuracy on the test set was 66% compared to the declared accuracy of about 89%. Later on, we found that we didn't use LSTM correctly when combined with the policy module, which is not shown in the report. After the correction, the highest accuracy increased to 77%.

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

## Report
The report can be found at [ReproductionReport.pdf](ReproductionReport.pdf).
