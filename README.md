This repository contains code, data and results for a research article entitled:

"Predicting responses to tweets from the official @USNavy account"

1 University of Central Florida, Department of Industrial Engineering & Management Systems, Orlando, Florida, USA </br>

In due course full description of usage will appear here.

The code is written in Python3 and requires GPU computing machine for achieving reasonable performance.

The whole repository is published under MIT License (please refer to the [License file](https://github.com/krzysztoffiok/predicting-response-to-tweets/blob/master/LICENSE)).

## Example results

TODO

## Installation if you wish to try our code:
Please clone this repository and carry out all computation yourself or extract zipped files downloaded from "release" section to use tweet embeddings computed in our work.

## How the code works:
You start with a data set of 4498 tweets (file is completely anonymized) which were previously filtered from all tweets pulished by the account in question after year 2016. The task is to predict response (likes, replies, retweets) to tweets based on unstructured tweet text and structured tweet data. Since precise prediction of response (regression) is an extremely difficult task, it was decided to address a simpler classification task with 3 classess aggregating the amount of response into "small", "moderate" and "high".

<strong>Steps for full reproduction of our results:</strong>

<strong>Step 1: prepare tweets</strong>
In order to carry out this step run prepare_tweets.ipynb. This file will filter, label and divide data into 5 cross validated folds.

<strong>Step 2: train Deep Learning Language Models and embed tweet text with trained models</strong>
Execute bash script by: "bash ./grid_train" to call many times model_train.py in a specified manner. This script will train all 4 models in all configurations (5 folds) and all 3 target variables, 60 training runs altogether.

Next, execute bash scripty by: "bash ./grid_embed" to call many times embed_sentences_flair.py in a specified manner. This script will create vector representations of tweet text (embeddings) by all trained language models in all folds and for all target variables. For training of Deep Learning Language Models(DLLMs) and embedding tweet text we utilize [Flair](https://github.com/flairNLP/flair).

<strong>Step 3: machine learning prediction of response to tweets</strong>
To repeat experiments carried out in our work please run bash scripts:
"bash ./grid_predict"
"bash ./grid_predict_mixed_features"
The first script uses machine_learning.py to carry out ML predictions for single language model features. The second script uses machine_learning.py to mix language model+structured features and do ML predictions.

<strong>Step 4: SHAP explanations</strong>
Our work utilizes [SHapley Additive exPlanations (SHAP)](https://github.com/slundberg/shap) for computing explanations of machine learning models.
If you wish to compute model explanations, please use the machine_learning.py file with commands described in the file, for example: "python3 machine_learning.py --target=likes --estimators=2 --language_model=structured  --ml_model=XBG --estimators=2 --mixed_features --shap_out"

## Acknowledgment
This research was carried out as part of the N000141812559 ONR research grant.

## Citation:<br/>
If you decide to use here published code or our dataset please cite our work in the following manner:
(please contact us directly at this time since the paper is still in preparation).
