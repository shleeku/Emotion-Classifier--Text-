from transformers import pipeline
from matplotlib import pyplot as plt
import pandas as pd
from datasets import load_dataset

# emotions = load_dataset("emotion")
# labels = emotions["train"].features["label"].names
# print(labels)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Change `transformersbook` to your Hub username
# model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
model_id = "transformer_model"
classifier = pipeline("text-classification", model=model_id)

custom_tweet = ["we found a great restaurant for lunch!"] #, "I am so sad"]
custom_tweet = ' '.join(custom_tweet)
print(custom_tweet)
preds = classifier(custom_tweet, return_all_scores=True)

preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()