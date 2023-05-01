import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import DistilBertTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import list_datasets
from datasets import load_dataset
import pandas as pd

# import sys
# print(sys.prefix)

# all_datasets = list_datasets()

# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")

emotions = load_dataset("emotion")

print(emotions['train'][0])

print(emotions.shape)

emotions.set_format(type="pandas")
print(emotions.shape)