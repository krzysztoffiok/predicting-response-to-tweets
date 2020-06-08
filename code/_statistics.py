import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn import metrics

targets = ["replies", "retweets", "likes"]
lms = ["roberta", "fasttext", "glove", "distilbert"]
classifiers = ["XGBClassifier", "RandomForestClassifier", "RidgeClassifier", "MLPClassifier"]


# chunking code borrowed from https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


# function to convert class labels from int to str
def decode_target_classes(x):
    x = str(x)
    if x == 0:
        return "small"
    elif x == 1:
        return "moderate"
    elif x == 2:
        return "high"


folds = list(chunks([x for x in range(4498)], 900))
for target in targets:
    f1FoldDict = dict()
    namesList = list()

    # compute per fold F1 scores for every variant with separate features
    df_preds = pd.read_excel(f"./results/{target}_all_250_False_predictions.xlsx")
    print(f"./results/{target}_all_250_False_trues.xlsx")
    df_trues = pd.read_excel(f"./results/{target}_all_250_False_trues.xlsx")
    df_preds.drop("Unnamed: 0", axis=1, inplace=True)
    cols = df_preds.columns
    for num, fold in enumerate(folds):
        trues = df_trues.iloc[fold][f"{target}_roberta_False_{classifiers[0]}"]
        for col in cols:
            if "RidgeClassifier" in col:
                df_preds.iloc[fold][col] = df_preds.iloc[fold][col].map(lambda x: decode_target_classes(x))
            else:
                preds = df_preds.iloc[fold][col]
            f1 = metrics.f1_score(y_true=trues, y_pred=preds, average="micro")
            f1FoldDict[f"{num}_{col}"] = f1
            namesList.append(col)

    # create per fold per variant F1 score data frame
    dfCompare = pd.DataFrame()
    for name in namesList:
        foldList = list()
        for i in range(5):
            foldList.append(f1FoldDict[f"{i}_{name}"])
        dfCompare[f"{name}"] = foldList

################################
    # compute per fold F1 scores for every variant with mixed features
    df_preds = pd.DataFrame()
    for lm in lms:
        df_preds = pd.concat([df_preds, pd.read_excel(f"./results/{target}_{lm}_250_True_predictions.xlsx")], axis=1)
        print(f"./results/{target}_{lm}_250_True_predictions.xlsx")
    df_trues = pd.read_excel(f"./results/{target}_roberta_250_True_trues.xlsx")
    df_preds.drop("Unnamed: 0", axis=1, inplace=True)
    cols = df_preds.columns
    for num, fold in enumerate(folds):
        trues = df_trues.iloc[fold][f"{target}_roberta_True_{classifiers[0]}"]
        for col in cols:
            if "RidgeClassifier" in col:
                df_preds.iloc[fold][col] = df_preds.iloc[fold][col].map(lambda x: decode_target_classes(x))
            else:
                preds = df_preds.iloc[fold][col]
            f1 = metrics.f1_score(y_true=trues, y_pred=preds, average="micro")
            f1FoldDict[f"{num}_{col}"] = f1
            namesList.append(col)

    # add entries to per fold per variant F1 score data frame
    for name in namesList:
        foldList = list()
        for i in range(5):
            foldList.append(f1FoldDict[f"{i}_{name}"])
        dfCompare[f"{name}"] = foldList
#######################################
    # start testing
    # normality test
    _normality = open(f"_normality.txt", "w")
    print(pg.normality(dfCompare), file=_normality)

    # ANOVA is computed here
    df_melt = pd.melt(dfCompare.reset_index(), id_vars=["index"], value_vars=namesList)
    df_melt.columns = ["index", "treatments", "value"]
    model = ols('value ~ C(treatments)', data=df_melt).fit()
    # print(model.summary())
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # pairways comparison
    tukey_ = open(f"./results/statistics/{target}_statistics.txt", "w+")

    if len(namesList) > 2:
        m_comp = pairwise_tukeyhsd(endog=df_melt['value'], groups=df_melt['treatments'], alpha=0.05)
        print(f"NORMALITY:", file=tukey_)
        print(pg.normality(dfCompare), file=tukey_)
        print("\n ANOVA:", file=tukey_)
        print(anova_table, file=tukey_)
        print("\n HSD Tukey:", file=tukey_)
        print(m_comp, file=tukey_)
        print(m_comp)
