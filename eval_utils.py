import json
import pathlib
from typing import Dict
import numpy as np
import spacy
from rouge_score import rouge_scorer
import re
import pandas as pd
import tqdm
from multiprocessing import Pool, Queue


def namedEntityRatio(nes_ori, nes_cand):
    nes_ori = set(nes_ori)

    if len(nes_ori) == 0:
        return 0
    if len(nes_cand) == 0:
        return 0

    nes_cand = set(nes_cand)

    recall = len(nes_ori.intersection(nes_cand)) / len(nes_ori)
    precision = len(nes_ori.intersection(nes_cand)) / len(nes_cand)
    if recall + precision == 0:
        return 0
    return 2 * (recall * precision) / (recall + precision)


def nerScore(original, generated, nlp):
    cost = np.zeros((len(original), len(generated)))
    named_entities_original = [
        list(map(lambda x: x.text, nlp(o).ents)) for o in original
    ]
    named_entities_candidate = [
        list(map(lambda x: x.text, nlp(o).ents)) for o in generated
    ]
    for i, o in enumerate(named_entities_original):
        for j, g in enumerate(named_entities_candidate):
            cost[i, j] = namedEntityRatio(o, g)

    return cost, named_entities_original, named_entities_candidate


def rougeScores(original, generated, scorer):
    rouge_scores = {}
    for statistic in ["fmeasure", "recall", "precision"]:
        cost_1 = np.zeros((len(original), len(generated)))
        cost_2 = np.zeros((len(original), len(generated)))
        cost_L = np.zeros((len(original), len(generated)))
        for i, o in enumerate(original):
            for j, g in enumerate(generated):
                scores = scorer.score(o, g)
                if statistic == "fmeasure":
                    cost_1[i, j] = scores["rouge1"].fmeasure
                    cost_2[i, j] = scores["rouge2"].fmeasure
                    cost_L[i, j] = scores["rougeL"].fmeasure
                elif statistic == "recall":
                    cost_1[i, j] = scores["rouge1"].recall
                    cost_2[i, j] = scores["rouge2"].recall
                    cost_L[i, j] = scores["rougeL"].recall
                elif statistic == "precision":
                    cost_1[i, j] = scores["rouge1"].precision
                    cost_2[i, j] = scores["rouge2"].precision
                    cost_L[i, j] = scores["rougeL"].precision
        rouge_scores[statistic] = {"rouge1": cost_1, "rouge2": cost_2, "rougeL": cost_L}

    return rouge_scores


def createScorers():
    nlp = spacy.load("en_core_web_sm")
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return {"nlp": nlp, "rouge": rouge}


def calcMetrics(ref, cand, scorers, strip=True):
    if strip:
        cand = ["".join(re.split(r'( *[\.\?!][\'"\)\]]* *)', c)[:2]) for c in cand]
    ner, ent_orig, ent_cand = nerScore(ref, cand, scorers["nlp"])
    df = pd.DataFrame(
        {
            "original": [ref],
            "candidate": [cand],
            "ner": [ner],
            "ent_orig": [ent_orig],
            "ent_cand": [ent_cand],
        }
    )

    rouge_scores = rougeScores(ref, cand, scorers["rouge"])
    for statistic in ["fmeasure", "recall", "precision"]:
        for rouge in ["1", "2", "L"]:
            df[f"rouge_{rouge}_{statistic}"] = [
                rouge_scores[statistic][f"rouge{rouge}"]
            ]

    return df


def getTopkMetrics(result, params, scorers, k=1):
    topk_data = [
        calcMetrics(result["original"], result["results"][i]["result"], scorers)
        for i in range(0, k)
        if len(result["results"]) > i
    ]
    for idx, metrics in enumerate(topk_data):
        for (k, v) in params.items():
            metrics[k] = v
        metrics["topk"] = idx
    topk_data = pd.concat(topk_data, ignore_index=True)
    return topk_data


def getAllResultsJson(dir: pathlib.Path):
    for child in dir.iterdir():
        if child.name == "result.json":
            params = child.parent / "params.json"
            params = json.load(params.open())
            results = json.load(child.open())
            if params["no_repeat_ngram_size"] is None:
                params["no_repeat_ngram_size"] = 0
            yield {"results": results, "params": params}
        elif child.is_dir():
            yield from getAllResultsJson(child)


def isNewRow(dataframe: pd.DataFrame, parameters: Dict):
    dataframe_columns = set(dataframe.columns)
    for (k, v) in parameters.items():
        if k not in dataframe_columns:
            return True
        dataframe = dataframe[dataframe[k] == v]
    return len(dataframe) == 0


def calculateAllNewMetrics(
    directory: pathlib.Path, existing: pd.DataFrame = None, topk=1
):
    all_results = getAllResultsJson(directory)
    if existing is not None:
        all_results = filter(lambda r: isNewRow(existing, r["params"]), all_results)
    else:
        existing = pd.DataFrame()

    scorers = createScorers()
    dataframes = []
    for result in tqdm.tqdm(list(all_results)):
        dataframes.append(
            getTopkMetrics(result["results"], result["params"], scorers, topk)
        )
    return dataframes


def getBestScores(dataframe: pd.DataFrame, metric: str):
    best_index = [np.unravel_index(np.argmax(v), v.shape) for v in dataframe[metric]]
    best_value = [np.max(v) for v in dataframe[metric]]
    return best_index, best_value


hyperparams = [
    "beamscore-method",
    "num_beams",
    "batchsize",
    "train_iteration",
    "beamgroups",
    "sample_num",
    "beamheight",
    "diversity",
    "randinit",
    "train_lr",
    "train_lrsched",
    "freqknown",
    "no_repeat_ngram_size",
    "dataset_size",
    "n_repetitions",
]


def keepOnlyBestTopk(
    dataframe: pd.DataFrame,
    metric="rouge_1_fmeasure_best_value",
    ascending=False,
    topk=1000,
):
    filtered_hyperparams = [h for h in hyperparams if h in dataframe.columns]
    split_df = dataframe.groupby(filtered_hyperparams)
    rows = []
    for (_, group) in split_df:
        group = group[group["topk"] <= topk]
        rows.append(group.sort_values(metric, ascending=ascending).head(1))
    return pd.concat(rows)

def keepOnlyBestTopkUniqueSentences(
    dataframe: pd.DataFrame,
    metric="rouge_1_fmeasure",
    ascending=False,
    topk=1000,
):
    filtered_hyperparams = [h for h in hyperparams if h in dataframe.columns]
    split_df = dataframe.groupby(filtered_hyperparams)
    rows = []
    for (_, group) in split_df:
        group = group[group["topk"] <= topk]
        split_group = group.groupby(f"{metric}_best_idx")
        for (_, subgroup) in split_group:
            rows.append(subgroup.sort_values(f"{metric}_best_value", ascending=ascending).head(1))
    return pd.concat(rows)

if __name__ == "__main__":
    print(calcMetrics(["The fat cat sits at home."], ["The cat"], createScorers()))

    a, b, c = nerScore(
        [
            "It is variable in form and may be fused dorsally with some of the thoracic segments or occasionally be in two parts, hinged dorsally."
        ],
        [
            "What is known about her is that she had three children: one of whom is unknown;the other two are known not to have been born;and one is not known to be living at all."
        ],
        createScorers()["nlp"],
    )

    print(a, b, c)
