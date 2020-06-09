import pandas as pd
import argparse
import numpy as np
# import flair
# import torch
import datatable
import time
import math

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

# prepare df for output to csv
# batch size for embedding tweet instances
bs = 32
tweets_to_embed = data['sentence'].copy()
print("beginning embedding")

# prepare mini batches
low_limits = list()
for x in range(0, len(tweets_to_embed), bs):
    low_limits.append(x)
up_limits = [x + bs for x in low_limits[:-1]]
up_limits.append(len(tweets_to_embed))

# a placeholder for embedded tweets and time of computation
newEmbedings = list()
embedding_times = list()

# embeddings tweets
for i in range(len(low_limits)):
    it = time.time()
    print(f"batch {math.ceil(up_limits[i] / bs)}")
    # get the list of current tweet instances
    slist = tweets_to_embed.iloc[low_limits[i]:up_limits[i]].to_list()

    # create a list of Sentence objects
    sentlist = list()
    for sent in slist:
        sentlist.append(Sentence(sent, use_tokenizer=True))

    # feed the list of Sentence objects to the model and output embeddings
    document_embeddings.embed(sentlist)

    # add embeddings of sentences to a new data frame
    for num, sentence in enumerate(sentlist):
        sent_emb = sentence.get_embedding()
        newEmbedings.append(sent_emb.squeeze().tolist())

    ft = time.time()
    embedding_times.append((ft - it) / bs)

print("Average tweet embedding time: ", np.array(embedding_times).mean())
print("Total tweet embedding time: ", len(tweets_to_embed)*np.array(embedding_times).mean())
# save all embeddings in a DataFrame
df = pd.DataFrame(newEmbedings)

# add rows with target variable and dummy_id for identification of rows
df = df.astype(np.float16)
df["row"] = data["row"]

print(df.head())

df.to_csv(f"./data/embeddings/{model_type}_{test_run}_encoded_sentences_{fold}.csv")

