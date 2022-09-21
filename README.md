# Semi-Supervised Text Classification with Dual Pseudo Supervision



## Dataset

we only sample a small part of data for submission, the complete data can be downloaded through the following link：

- AGNews : https://pytorch.org/text/stable/datasets.html#ag-news
- Yelp：https://pytorch.org/text/stable/datasets.html#yelpreviewfull

## Usage

Train the model by 100 labeled data of AGNews dataset:

```
python main.py --dataset AGNews --num_labeled 100 --num_unlabeled 20000 --batch-size 32 --max_len 64 --teacher_lr 0.0001 
```

Train the model by 100 labeled data of Yelp dataset:

```
python main.py --dataset Yelp --num_labeled 100 --num_unlabeled 20000 --batch-size 4 --max_len 256 --threshold 0.95 --temperature 0.5 --drop 0.3
```

We can change the parameters --num_labeled and --num_unlabeled to achieve the training result that we want.

Monitoring training progress :

```
tensorboard --logdir results
```

## Requirements
- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- wandb
- numpy
- tqdm
- sklearn
- Transformers
