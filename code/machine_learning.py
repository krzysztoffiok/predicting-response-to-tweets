import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datatable
from sklearn import metrics
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb
import shap
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

"""
Example use:
To output SHAP explanations for a given model
python3 machine_learning.py --target=likes --estimators=250 --language_model=roberta  --ml_model=XGB --shap_out --mixed_features

To output cross validated model predictions
python3 machine_learning.py --target=likes --estimators=2 --language_model=glove  --ml_model=all 

To output cross validated model predictions with mixed features (features from language model + structured)
python3 machine_learning.py --target=likes --estimators=2 --language_model=glove  --ml_model=all --mixed_features

--target: defines target variable
--estimators: number of estimators in machine learning classification models
--language_model: feature set provided by selected language model
--ml_model: machine learning classification model
--shap_out: enables only SHAP explanations for the selected ml_model and fold instead of cross validated tests
--mixed_features: enables to add structured features to a selected language model

To repeat experiments carried out in our work use two bash scripts:
bash ./grid_predict
bash ./grid_predict_mixed_features
"""


parser = argparse.ArgumentParser(description='Carry out machine learning')

parser.add_argument('--target', required=True, type=str, default='retweets', help="possible options:"
                                                                                  "likes, retweets, replies")
parser.add_argument('--language_model', required=True, type=str, default='glove',
                    help="possible options: fasttext, glove, distilbert, roberta, structured, all")
parser.add_argument('--estimators', required=False, type=int, default=250,
                    help="number of estimators in machine learning classification models")
parser.add_argument('--fold', required=False, type=int, default=0, help="choose fold in range: 0-4")
parser.add_argument('--shap_out', required=False, default=False, action='store_true', help='output SHAP explanations')
parser.add_argument('--ml_model', required=False, type=str, default='RF', help="possible options: RF, XGB, Ridge, all")
parser.add_argument('--mixed_features', required=False, action='store_true', default=False,
                    help="enables to add structured features to a selected language model")

args = parser.parse_args()
_target = args.target
_estimators = args.estimators
_fold = args.fold
_language_model = args.language_model
_ml_model = args.ml_model
_shap_out = args.shap_out
_mixed_features = args.mixed_features

# read source data
source_data = datatable.fread("./data/from_USNavy_for_flair.csv").to_pandas()

# define which classification models to use
ml_model_selector = {
    "XGB": xgb.XGBClassifier(objective='multi:softprob', n_jobs=24, learning_rate=0.03,
                             max_depth=10, subsample=0.7, colsample_bytree=0.6,
                             random_state=2020, n_estimators=_estimators),
    "RF": RandomForestClassifier(n_estimators=_estimators, max_depth=7, min_samples_split=2,
                                 min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020),
    "Ridge": RidgeClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=2000)
}

# define timestamp as index
source_data['timestamp'] = pd.to_datetime(source_data['timestamp'], errors='coerce')

# display percentage of selected boolean data features
source_data["has_hash"] = source_data["hashtags"].astype(str).str.len() > 4
source_data["has_url"] = source_data["links"].astype(str).str.len() > 4
source_data["has_url2"] = source_data["sentence"].str.contains("URL") == True
llen = len(source_data)
source_data["has_retweet"] = source_data["sentence"].str.contains("_RETWEET") == True
print("Has links: ", round(source_data["has_url2"].astype(int).sum()/llen*100, 2))
print("Has images: ", round(source_data["has_media"].astype(int).sum()/llen*100, 2))
print("Includes retweets: ", round(source_data["has_retweet"].astype(int).sum()/llen*100, 2))
print("Has hashtags: ", round(source_data["has_hash"].astype(int).sum()/llen*100, 2))
print("Is a reply: ", round(source_data["is_reply_to"].astype(int).sum()/llen*100, 2))

""" Very limited feature engineering """
# creating a year-month feature indicating number of months counted from the 1st data instance
source_data["ym"] = (((source_data['timestamp'].dt.year.astype(str).str[3:]).astype(int)-7)*12) +\
                    source_data["timestamp"].dt.month.astype(int)

# list of utlized structured tweet features
structured_features = ['has_media', 'has_hash', "has_url2", 'is_reply_to', 'has_retweet', "ym"]

# remove duplicated columns
source_data = source_data.loc[:, ~source_data.columns.duplicated()]

# 5 fold CV
# setup random state
np.random.seed(13)

fold_number = 5
kf = KFold(n_splits=fold_number, random_state=13, shuffle=True)

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = {}
val_indexes = {}
test_indexes = {}

# train sets for Machine Learning
train_ml = {}
i = 0

# this split (with fold_number=5) results in: 20% test, 10% val, 70% train for Flair framework
# and the same 20% test and 80 % train for Machine Learning
indexes = list(range(0, len(source_data)))
for train_index, test_index in kf.split(indexes):
    test_indexes[i] = test_index
    train_ml[i] = train_index
    train_index, val_index = train_test_split(train_index, test_size=0.125, random_state=13, shuffle=True)
    train_indexes[i] = train_index
    val_indexes[i] = val_index
    i += 1

# test sets for Machine Learning are equal to those for Flair framework
test_ml = test_indexes

# define columns with target variables
target_columns = [_target]

# define which embedding files to read
if _language_model == "all":
    embeddings = ["fasttext", "glove", "distilbert", "roberta"]
else:
    embeddings = [_language_model]
# instantiate list of data frames with features and a list of feature names for each df
dfemblist = []

# Initialize a dictionary with all features used later on in Machine Learning
allFeatures = {}

# read embedding files and define corresponding feature names (lists of names)
if "structured" not in embeddings:
    for embedding in embeddings:
        for target in target_columns:
            embfeaturedict = {}
            for fold in range(fold_number):
                # read encoded sentences by the selected language model
                dfemb = datatable.fread(f"./data/embeddings/{target}_{embedding}_encoded_sentences_{fold}.csv")\
                    .to_pandas()

                print("loaded file: ", f"./data/embeddings/{target}_{embedding}_encoded_sentences_{fold}.csv")
                dfemb.drop(["row"], axis=1, inplace=True)
                # create a new list of feature names in dfemb
                embfeatures = [f"{target}{embedding}{fold}row"]

                # define number of feature columns (columns - 1)
                number_of_feature_columns = len(dfemb.columns) - 1

                # create unique feature (column) names
                embfeatures.extend([f"{target}{embedding}{fold}_{x}" for x in range(number_of_feature_columns)])

                # change feature names to new ones
                dfemb.columns = embfeatures

                # append features from each language model and target variable
                embfeaturedict[fold] = [f"{target}{embedding}{fold}_{x}" for x in range(number_of_feature_columns)]

                # append encoded sentences by the selected language model to a list of data frames
                dfemblist.append(dfemb)

            # create entry in dictionary with all features for each trained language model
            allFeatures[f"{target}_{embedding}"] = embfeaturedict


# add the structured key to dictionary with allFeatures from various language models
if _language_model == "all" or _language_model == "structured" or _mixed_features:
    # Create per-fold feature dictionary for structured tweet features
    foldStructuredfeatures = {}
    for fold, rows in train_ml.items():
        foldStructuredfeatures[fold] = structured_features.copy()

    for target in target_columns:
        allFeatures[f"{target}_structured"] = foldStructuredfeatures

# concat all Data Frames into one df_ml that will be used in Machine Learning
print("merging data frames")
dftemp = source_data.copy()
for dfemb in dfemblist:
    dftemp = pd.concat([dftemp, dfemb], axis=1)
df_ml = dftemp
print("done")

# Define separate lists of names of trained and not trained language models that can be tested
trained_LMs = []
for target in target_columns:
    trained_LMs.extend([f"{target}_{x}" for x in embeddings])
all_language_models = trained_LMs.copy()

if _language_model == "all":
    explainable_LMs = [f"{x}_structured" for x in target_columns]
    all_language_models.extend(explainable_LMs)


# function that trains
def ML_classification(classification_model, language_model, target):
    """
    Function to train classification models on features provided by language models
    Example use: classification_model=RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020)
                language_model=
    possible options for language model list are:
    """
    # list of analyzed language models
    model = classification_model
    print(type(model).__name__)
    preds = []
    trues = []
    if type(model).__name__ != "RidgeClassifier":
        df_ml["target_ml"] = df_ml[f"{target}"]
    else:
        df_ml["target_ml"] = df_ml[f"ridge{target}"]

    # for each fold
    for fold in range(fold_number):
        # chose appropriate features and data
        features = allFeatures[language_model][fold]
        if _mixed_features:
            features.extend(allFeatures[f"{target}_structured"][fold])
        features = set(features)
        train_index = train_ml[fold]
        test_index = test_ml[fold]

        train_data = df_ml[features].iloc[train_index]
        target_train_data = df_ml["target_ml"].iloc[train_index]
        test_data = df_ml[features].iloc[test_index]
        target_test_data = df_ml.iloc[test_index]["target_ml"]
        model.fit(train_data, target_train_data)

        preds.append(model.predict(test_data).tolist())
        trues.append(target_test_data.tolist())

    if _mixed_features:
        print(language_model, f"{target}_structured")
    else:
        print(language_model)
    mcc = metrics.matthews_corrcoef(y_true=sum(trues, []), y_pred=sum(preds, []))
    f1 = metrics.f1_score(y_true=sum(trues, []), y_pred=sum(preds, []), average="micro")
    print("MCC: ", round(mcc, 3))
    print("F1: ", round(f1, 3))
    return sum(preds, []), sum(trues, []), round(f1, 3), round(mcc, 3)


# define target
target = _target

# instantiate dictionary for data frames with results
allPreds = {}
allTrues = {}

if _ml_model == "all":
    models = []
    for key in ml_model_selector.keys():
        models.append(ml_model_selector[key])
else:
    models = [ml_model_selector[_ml_model]]

if not _shap_out:
    # prepare dictionaries for metric scores
    f1dict = {}
    mccdict = {}

    # use features from selected language models
    for language_model in all_language_models:
        if target not in language_model:
            continue
        else:
            # for training of selected classification models
            for classification_model in models:

                # linear model requires to operate on floats, not strings
                if "RidgeClassifier" in type(classification_model).__name__:
                    def encode_target_classes(x):
                        x = str(x)
                        if x == "small":
                            return 0
                        elif x == "moderate":
                            return 1
                        elif x == "high":
                            return 2
                    df_ml[f"ridge{target}"] = df_ml[target].map(lambda x: encode_target_classes(x))

                # train model
                preds, trues, f1, mcc = ML_classification(classification_model, language_model, target)

                # store metric scores
                f1dict[f"{language_model}_{_mixed_features}_{type(classification_model).__name__}"] = f1
                mccdict[f"{language_model}_{_mixed_features}_{type(classification_model).__name__}"] = mcc

                # store model predictions
                allPreds[f"{language_model}_{_mixed_features}_{type(classification_model).__name__}"] = preds.copy()
                allTrues[f"{language_model}_{_mixed_features}_{type(classification_model).__name__}"] = trues.copy()

    # save model predictions together with true sentiment labels
    pd.DataFrame(allPreds).to_excel(f"./results/{target}_{_language_model}"
                                    f"_{_estimators}_{_mixed_features}_predictions.xlsx")
    pd.DataFrame(allTrues).to_excel(f"./results/{target}_{_language_model}_{_estimators}_{_mixed_features}_trues.xlsx")
    # save metric scores
    pd.DataFrame(f1dict, index=[0]).to_excel\
        (f"./results/{target}_{_language_model}_{_estimators}_{_mixed_features}_f1.xlsx")
    pd.DataFrame(mccdict, index=[0]).to_excel\
        (f"./results/{target}_{_language_model}_{_estimators}_{_mixed_features}_mcc.xlsx")

# if SHAP explanations are desired
# this code selects the 1st from list of provided classification models for explanations

# feature set model for SHAP explanations
_shap = f"{_target}_{_language_model}"

if _shap_out:
    if not type(models[0]).__name__ == "RidgeClassifier":
        def train_model_for_shap(classification_model, language_model, fold, target):
            """
            Function to train a single Language Model for SHAP explanations
            Example use: classification_model=RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2,
                                     min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020),
                        language_model="Term Frequency",
                        fold = 2
            possible options for language model are: "Term Frequency" or "LIWC".
            possible fold values: 0, 1, 2, 3, 4
            """
            # list of analyzed language models
            language_model = language_model
            fold = fold
            model = classification_model
            print(type(model).__name__)
            results = {}
            names = []

            df_ml["target_ml"] = df_ml[f"{target}"]

            features = allFeatures[language_model][fold]
            if _mixed_features:
                features.extend(allFeatures[f"{target}_structured"][fold])
            features = set(features)
            preds = []
            trues = []

            train_index = train_ml[fold]
            test_index = test_ml[fold]

            train_data = df_ml[features].iloc[train_index]
            target_train_data = df_ml["target_ml"].iloc[train_index]
            test_data = df_ml[features].iloc[test_index]
            target_test_data = df_ml.iloc[test_index]["target_ml"]
            model.fit(train_data, target_train_data)

            preds.append(model.predict(test_data).tolist())
            trues.append(target_test_data.tolist())

            if _mixed_features:
                print(language_model, f"{target}_structured")
            else:
                print(language_model)
            mcc = metrics.matthews_corrcoef(y_true=sum(trues, []), y_pred=sum(preds, []))
            f1 = metrics.f1_score(y_true=sum(trues, []), y_pred=sum(preds, []), average="weighted")
            print("MCC: ", round(mcc, 3))
            print("F1: ", round(f1, 3))
            return model, train_data, test_data


        def explain_model(model, train_data, test_data):
            """
            Function that computes and displays SHAP model explanations
            """
            model_name = type(shap_model).__name__
            random.seed(13)
            samples_to_explain = 1000
            if model_name not in ["RandomForestClassifier", "XGBClassifier"]:
                explainer = shap.KernelExplainer(model.predict_proba, train_data[:50], link="identity")
                shap_values = explainer.shap_values(train_data[:50], nsamples=samples_to_explain,
                                                    l1_reg="num_features(100)")
            else:
                explainer = shap.TreeExplainer(model, feature_perturbation=150)
                shap_values = explainer.shap_values(train_data)

            fig = shap.summary_plot(shap_values, test_data, max_display=15, show=False, plot_size=(20, 13))
            return fig


        # prepare model for SHAP explanations
        print("Computing model for SHAP explanations")
        shap_model, train_data, test_data = train_model_for_shap(
            classification_model=models[0],
            language_model=_shap,
            fold=_fold, target=_target)

        fig = explain_model(model=shap_model, train_data=train_data, test_data=test_data)
        plt.savefig(f'./results/{_target}_{_shap}_{_fold}_{_estimators}_{_mixed_features}_{_ml_model}_summary_plot.png')
    else:
        print("RidgeClassifier is not yet supported in our SHAP explanations.")
