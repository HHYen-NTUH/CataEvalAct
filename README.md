# CataEvalAct

## Preparation

1. Create conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate cataevalact
```

2.If you wish not to train your own model, you can download our [pre-trained checkpoints](https://fethb3053wy-my.sharepoint.com/:f:/g/personal/chiahungyang_stmedical_tw/IgAdWOmjxAc9Tqcjf9X0KqlyAeXb8iHo6DdfJSGHuJkX0nY?e=vgbBES) into `pretrained` and modify the bash scripts to use the `pretrained` directory instead of the `model` directory.

## Pre-training

Run

```bash
bash experiment/videomae/ccc/pretrain.sh
bash experiment/videomae/phaco/pretrain.sh
```

## Training

Run

```bash
bash experiment/videomae/ccc/train.sh
bash experiment/videomae/phaco/train.sh
```

## Evaluation

Run

```bash
bash experiment/videomae/ccc/evaluate.sh
bash experiment/videomae/phaco/evaluate.sh
bash experiment/videomae/evaluate.sh
```
