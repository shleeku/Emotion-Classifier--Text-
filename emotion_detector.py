from datasets import list_datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import DistilBertTokenizer
from transformers import AutoModelForSequenceClassification

# import sys
# print(sys.prefix)

all_datasets = list_datasets()

# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")

emotions = load_dataset("emotion")

# print(emotions)

# dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1"
# emotions_remote = load_dataset("csv", data_files=dataset_url, sep=";",
#                                names=["text", "label"])

emotions.set_format(type="pandas")
df = emotions["train"][:]
# print(df.head())

# print(emotions["train"].features["label"].int2str(0))
# print(df["label"][0:5])

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)

# print(df.head())

# df["label_name"].value_counts(ascending=True).plot.barh()
# plt.title("Frequency of Classes")
# plt.show()

# print(df["text"].str.split()[0:5])

df["Words Per Tweet"] = df["text"].str.split().apply(len)
# df.boxplot("Words Per Tweet", by="label_name", grid=False,
#            showfliers=False, color="black")
# plt.suptitle("")
# plt.xlabel("")
# plt.show()

emotions.reset_format()

text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
# print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
# print(input_ids)

categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})

# print(pd.get_dummies(categorical_df["Name"]))

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
# print(one_hot_encodings.shape)

# print(f"Token: {tokenized_text[0]}")
# print(f"Tensor index: {input_ids[0]}")
# print(f"One-hot: {one_hot_encodings[0]}")

tokenized_text = text.split()
# print(tokenized_text)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
# print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print(tokens)
# print(tokenizer.convert_tokens_to_string(tokens))
# print(tokenizer.vocab_size)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# print(emotions_encoded["train"].column_names)
# print(emotions_encoded["train"][0])

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "I'm so angry!"

# tokens = tokenizer(text)
# tokens = tokenizer.convert_ids_to_tokens(tokens.input_ids)
# print(tokens)

# print(tokenizer(text))
# print(tokenizer(text, return_tensors="pt"))

inputs = tokenizer(text, return_tensors="pt")

# print(f"Input tensor shape: {inputs['input_ids'].size()}")

# print(inputs.items())

# for k,v in inputs.items():
#     print("k: ", k)
#     print("v: ", v)

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
# print(outputs)

# print(outputs.last_hidden_state.size())


def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return{"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# print(emotions_encoded["train"].shape)

#

# torch.cuda.empty_cache()

# emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True, batch_size=500)

# print(emotions["train"].features)

num_labels = 6
model = (AutoModelForSequenceClassification)