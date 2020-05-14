import pandas as pd
import argparse
import numpy as np
# import flair
# import torch
import datatable

""" example use to create tweet-level embeddings

python3 embed_sentences_flair.py  --test_run=fasttext --model=retweets

"""
# flair.device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Embed tweet data')
parser.add_argument('--test_run', required=True, default="fasttext",
                    type=str, help='name of model')
parser.add_argument("--nrows", required=False, default=6066, type=int)
parser.add_argument("--model", required=True, default="retweets", type=str)
parser.add_argument("--fold", required=False, default=5, type=int)

args = parser.parse_args()

fold = args.fold
test_run = args.test_run
nrows = args.nrows
model_type = args.model

# read data
df = datatable.fread(f"./data/from_USNavy_for_flair.csv").to_pandas()
print(len(df))
df = df.head(nrows)
data = df.copy()

data = data[['sentence', "row"]]

# load classifier
from flair.models import TextClassifier
from flair.data import Sentence
model = TextClassifier.load(f'./data/model_{model_type}_{fold}/_{test_run}_best-model.pt')
document_embeddings = model.document_embeddings

# prepare df for output csv
hidden_size = document_embeddings.embedding_length
columns = [x for x in range(hidden_size)]
df = pd.DataFrame(columns=[columns])

for num, sent in enumerate(data['sentence']):
    # prepare each tweet for flair tokenized format
    sentence = Sentence(sent, use_tokenizer=True)
    # embed sentence
    document_embeddings.embed(sentence)
    sent_emb = sentence.get_embedding()
    # add new row to df
    df.loc[num] = sent_emb.squeeze().tolist()

# add rows with target variable and dummy_id for identification of rows
df = df.astype(np.float16)
df["row"] = data["row"]

print(df.head())

df.to_csv(f"./data/embeddings/{model_type}_{test_run}_encoded_sentences_{fold}.csv")

