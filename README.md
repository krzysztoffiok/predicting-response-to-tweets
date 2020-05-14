This repository contains code, data and results for a research article entitled:

"Predicting responses to tweets from the official @USNavy account"

1 University of Central Florida, Department of Industrial Engineering & Management Systems, Orlando, Florida, USA </br>

In due course full description of usage will appear here.

The code is written in Python3 and requires GPU computing machine for achieving reasonable performance.

The whole repository is published under MIT License (please refer to the [License file](https://github.com/krzysztoffiok/predicting-response-to-tweets/blob/master/LICENSE)).

## Example results

Mean F1 micro scores for 3 target variables: Replies, Likes, Retweets, for 3 machine learning classifiers: Ridge (R), Random Forest (RF) and XGBoost (GB), for structured features (feature group I), features extracted by Deep Learning Feature Extraction (feature group II) carried out with use of [Flair framework](https://github.com/flairNLP/flair)  with selected Langualge Models: Glove (GL), FastText (FT), DistiLBERT (DB) and RoBERTa (RB) and and union of both feature types (feature group III).


Feature Group | Feature Set | ML classifier | Replies | Likes | Retweets
-- | -- | -- | -- | -- | --
I | S | GB | 0.5 | 0.558 | 0.532
I | S | RF | 0.502 | 0.572 | 0.54
I | S | R | 0.489 | 0.523 | 0.522
II | FT | GB | 0.533 | 0.592 | 0.611
II | FT | RF | 0.541 | 0.592 | 0.606
II | FT | R | 0.542 | 0.589 | 0.611
II | GL | GB | 0.534 | 0.604 | 0.61
II | GL | RF | 0.537 | 0.618 | 0.604
II | GL | R | 0.526 | 0.602 | 0.601
II | DB | GB | 0.553 | 0.62 | 0.613
II | DB | RF | 0.558 | 0.628 | 0.61
II | DB | R | 0.55 | 0.615 | 0.613
II | RB | GB | 0.516 | 0.641 | 0.64
II | RB | RF | 0.531 | 0.631 | 0.631
II | RB | R | 0.526 | 0.626 | 0.627
III | SFT | GB | 0.541 | 0.593 | 0.618
III | SFT | RF | 0.54 | 0.593 | 0.607
III | SFT | R | 0.546 | 0.596 | 0.616
III | SGL | GB | 0.537 | 0.606 | 0.611
III | SGL | RF | 0.54 | 0.616 | 0.604
III | SGL | R | 0.532 | 0.61 | 0.609
III | SDB | GB | 0.552 | 0.618 | 0.611
III | SDB | RF | 0.557 | 0.626 | 0.609
III | SDB | R | 0.556 | 0.622 | 0.612
III | SRB | GB | 0.537 | 0.655 | 0.65
III | SRB | RF | 0.531 | 0.631 | 0.637
III | SRB | R | 0.536 | 0.633 | 0.637



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
The first script uses machine_learning.py to carry out ML predictions for single language model features. The second script uses machine_learning.py to mix language model+structured features and do ML predictions. Computed predictions and metric scores are stored in results folder.

<strong>Step 4: SHAP explanations</strong>
Our work utilizes [SHapley Additive exPlanations (SHAP)](https://github.com/slundberg/shap) for computing explanations of machine learning models.
If you wish to compute model explanations, please use the machine_learning.py file with commands described in the file, for example: "python3 machine_learning.py --target=likes --estimators=2 --language_model=structured  --ml_model=XBG --estimators=2 --mixed_features --shap_out"

## Acknowledgment
This research was carried out as part of the N000141812559 ONR research grant.

## Citation:<br/>
If you decide to use here published code or our dataset please cite our work in the following manner:
(please contact us directly at this time since the paper is still in preparation).
