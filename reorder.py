# python reorder_single.py --bs $BS --alpha $alpha --insert

import argparse
import copy
import gc
import itertools
import json
import random
import sys
from itertools import permutations
from os import sep

import numpy as np
import pandas as pd
import torch
import torch as T
from torch.autograd import grad

import wandb
from dlg_utils import calculateGradLoss, calculatePerplexity, get_grad_gpt2
from eval_utils import eval

stride = 16


def shard_sequence_using_eos(input, sep_token_id=50256):
    ret = []
    start = 0
    for i in range(len(input)):
        if input[i] == sep_token_id:
            ret.append(input[start:i])
            start = i + 1
    ret.append(input[start:])
    return ret


def score(
    model, padded, padded_s, original_dy_dx, tokenizer_GPT, length=20, batchsize=1
):
    padded = padded.long()

    gradscore = []

    for idx, s in enumerate(padded_s):
        if batchsize == 1:
            s = s.replace("<|endoftext|>", "")
        else:
            p = shard_sequence_using_eos(padded[idx][1:-1])
            s = tokenizer_GPT.batch_decode(p)  # Seperate into sentences of size
        padded_dy_dx = get_grad_gpt2(s, tokenizer_GPT, model)
        gradnorm = torch.norm(padded_dy_dx[LAYER_IDX]).cpu().numpy().item()
        ppl = calculatePerplexity(s, model, tokenizer_GPT).item()

        gradscore.append(gradnorm + ALPHA * ppl)
    gradscore = np.asarray(gradscore)

    wordscoress, scoress, predss = [], [], []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride : min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = T.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride : min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .cpu()
            .detach()
        )
        scores = wordscores.sum(1)

        wordscoress.append(wordscores)
        scoress.append(scores)
        predss.append(preds.cpu().detach())

    wordscores = T.cat(wordscoress)
    scores = T.cat(scoress)
    preds = T.cat(predss)

    return gradscore, scores, wordscores, -preds


cand_orders = {
    3: [
        list(ll)
        for ll in list(permutations(range(1, 5)))
        if ll[0] == 1 and ll[-1] == 4 and ll != tuple(range(1, 5))
    ],
    4: [
        list(ll)
        for ll in list(permutations(range(1, 6)))
        if ll[0] == 1 and ll[-1] == 5 and ll != tuple(range(1, 6))
    ],
    5: [
        list(ll)
        for ll in list(permutations(range(1, 7)))
        if ll[0] == 1 and ll[-1] == 6 and ll != tuple(range(1, 7))
    ],
    6: [
        list(ll)
        for ll in list(permutations(range(1, 8)))
        if ll[0] == 1 and ll[-1] == 6 and ll != tuple(range(1, 8))
    ],
    7: [
        list(ll)
        for ll in list(permutations(range(1, 9)))
        if ll[0] == 1 and ll[-1] == 6 and ll != tuple(range(1, 9))
    ],
}


class LocalSwap:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def computeStats(self, curr_tokens, original_dy_dx, do_split=False):

        generated_sentence = self.tokenizer.decode(curr_tokens)
        if do_split:
            generated_sentence = generated_sentence.split("<|endoftext|>")

        generated_dy_dx = get_grad_gpt2(generated_sentence, self.tokenizer, self.model)

        gradnorm = torch.norm(generated_dy_dx[LAYER_IDX]).cpu().numpy().item()
        ppl = calculatePerplexity(
            generated_sentence[0], self.model, self.tokenizer
        ).item()
        return {
            "tokens": curr_tokens,
            "sentence": generated_sentence,
            "gradloss_mse": gradnorm + ALPHA * ppl,
        }

    def createAllOrderings(
        self, sentence_size=40, without_order=False, max_depth=5, sep_idx=None
    ):
        for d in range(1, max_depth):
            tuples = itertools.product(range(1, sentence_size), range(1, sentence_size))
            if sep_idx == []:
                tuples = (
                    t for t in tuples if t[0] != t[1]
                )  # Only swap inside the first sentence! IMPORTANT!
                # if t[0] != t[1] and (t[0] not in sep_idx) and (t[1] not in sep_idx)
            else:
                tuples = (
                    t
                    for t in tuples
                    if t[0] != t[1]
                    and (t[0] < sep_idx[0])
                    and (
                        t[1] < sep_idx[0]
                    )  # Only swap inside the first sentence! IMPORTANT!
                    # if t[0] != t[1] and (t[0] not in sep_idx) and (t[1] not in sep_idx)
                )
            if without_order:
                tuples = (t for t in tuples if t[0] < t[1])
            tuples = list(tuples)

            tuple_list = [copy.copy(tuples) for _ in range(d)]

            for t in tuple_list:
                random.shuffle(t)

            for ordering in itertools.product(*tuple_list):
                yield ordering

    def reorderAlgorithm(self, original_dy_dx, tokens, generator, n=50):
        values = []
        for _, sequence in zip(range(n), generator):
            curr_tokens = copy.copy(tokens)
            for (i, j) in sequence:
                curr_tokens[i], curr_tokens[j] = curr_tokens[j], curr_tokens[i]
            values.append(self.computeStats(curr_tokens, original_dy_dx, do_split=True))

        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])

    def insertAlgorithm(
        self, original_dy_dx, tokens, token_pool, generator, n=50
    ):  # insert the i-th token from the pool at position
        if generator == None:  # insertion disabled
            return {"gradloss_mse": 10e100}
        if len(tokens) >= 40:  # disable insertion if the sentence is alreday very long
            return {"gradloss_mse": 10e100}
        values = []
        for _, sequence in zip(range(n), generator):
            curr_tokens = copy.copy(tokens)

            for (i, j) in sequence:
                # if i < j:
                #     curr_tokens = (
                #         curr_tokens[0:i]
                #         + curr_tokens[i + 1 : j + 1]
                #         + [curr_tokens[i]]
                #         + curr_tokens[j + 1 :]
                #     )
                # else:
                if i < len(token_pool) and j < len(tokens):
                    curr_tokens = curr_tokens[0:j] + [token_pool[i]] + curr_tokens[j:]
            values.append(self.computeStats(curr_tokens, original_dy_dx, do_split=True))

        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])


def shuffle_proposals_rand(length, bs, kopt):
    L = length
    orders = []
    o = np.arange(2, kopt + 1)
    np.random.shuffle(o)
    o = np.insert(o, 0, 1)
    o = np.insert(o, len(o), kopt + 1)
    orders = [o for _ in range(bs)]

    imod = [(torch.randperm(L - 2) + 1)[:kopt].sort()[0] for _ in range(bs)]
    topv = torch.zeros(bs)

    return T.stack(imod), topv, orders


def shuffle_proposals(mat, topk, bs, kopt):

    L = mat.shape[0]

    I = T.zeros((kopt,) + (L,) * (kopt)).long()
    for i in range(kopt):
        I[i] = T.arange(L).view((-1,) + (1,) * (kopt - 1 - i))

    mask = 0 < I[0]
    for i in range(kopt - 1):
        mask *= I[i] < I[i + 1]

    lv = mat.view(-1)

    orders = cand_orders[kopt]

    o = np.array(orders[np.random.randint(len(orders))])
    then = T.zeros((L,) * kopt)
    now = T.zeros_like(then)
    for i in range(kopt):
        now += lv[L * I[i] + I[i]]
        then += lv[L * I[o[i] - 1] + I[o[i + 1] - 2]]

    A = then - now

    A[~mask] = -1001

    topv, topi = A.view(-1).topk(min(A.numel(), topk))
    indices = np.random.randint(topi.shape[0], size=(bs,))
    topv = topv[indices]
    topi = topi[indices]

    orders = [o] * bs

    imod = [(topi // L ** (kopt - 1 - i)) % L for i in range(kopt)]

    return T.stack(imod, -1), topv, orders


def ibis(
    model,
    before,
    sentence,
    after,
    bs,
    topk,
    its,
    patience,
    warminit=False,
    gluemask=None,
    original_dy_dx=None,
    tokenizer=None,
    score_fn="nll",
    batchsize=1,
    original_sentence=None,
):
    device = model.device
    sent = sentence

    padded = T.cat([before, sent, after], 0).unsqueeze(0).to(device)
    padded_s = tokenizer.batch_decode(padded)

    gradloss, sc, wsc, spr = score(
        model, padded, padded_s, original_dy_dx, tokenizer, batchsize=1
    )  # Reorder single sentence!

    if score_fn == "grad":
        orscore = gradloss
    else:
        orscore = sc[0]
    yield orscore

    bestscore = orscore

    bestsc = spr[0]

    lfix, rfix, blanks = before.shape[0] - 1, after.shape[0] - 1, 0

    permsents = [
        T.cat([before, T.from_numpy((sent.numpy())), after], 0) for _ in range(bs)
    ]

    bestmask = np.full(permsents[0].shape, True)

    if gluemask is not None:
        bestmask[lfix + 1 : -rfix - 1] = gluemask

    permmasks = [bestmask.copy() for _ in range(bs)]

    if not warminit:
        seg = list(np.nonzero(bestmask[lfix + 1 : -rfix - 1])[0]) + [len(sent)]
        for b in range(bs):
            perm = np.random.permutation(len(seg) - 1)
            ns = []
            nm = []
            for i in range(len(seg) - 1):
                ns.append(sent[seg[perm[i]] : seg[perm[i] + 1]])
                nm.append(
                    bestmask[lfix + 1 : -rfix - 1][seg[perm[i]] : seg[perm[i] + 1]]
                )
            permsents[b][lfix + 1 : -rfix - 1] = T.cat(ns, 0)
            permmasks[b][lfix + 1 : -rfix - 1] = np.concatenate(nm, 0)

    padded = T.stack(permsents, 0).to(device)
    padded_s = tokenizer.batch_decode(padded)

    bestsent = np.zeros(padded[0].shape)

    bestscore = 1e20

    movetype = "init"
    nch = 0
    candidates = np.array([1] * bs)
    last_imp = 0
    for it in range(IBIS_MAX_STEP):

        gc.collect()

        if it - last_imp > patience:
            break

        gradloss, sc, wsc, spr = score(
            model,
            padded,
            padded_s,
            original_dy_dx,
            tokenizer,
            batchsize=batchsize,
        )
        if score_fn == "grad":
            sc = gradloss
        else:
            sc = sc.numpy()

        if it == 0:
            bestwsc = wsc[0]

        if sc.min() < bestscore:
            if it == 0 or np.any(permsents[sc.argmin()] != bestsent):

                nch += 1

                bestsent, bestscore, bgradloss, bestsc, bestwsc, bestmask = (
                    permsents[sc.argmin()],
                    sc.min(),
                    gradloss[sc.argmin()],
                    spr[sc.argmin()],
                    wsc[sc.argmin()],
                    permmasks[sc.argmin()],
                )
                recon_metrics = eval(
                    ori_single,
                    padded_s[sc.argmin()].split("<|endoftext|>")[1:-1],
                    model,
                    tokenizer,
                )
                wandb.log(
                    {
                        "Coarse: GradLoss": sc.min(),
                        "Coarse: BLEU": recon_metrics["bleu"],
                        "Coarse: NER": recon_metrics["ner"],
                        "Coarse: PPL": recon_metrics["ppl"],
                        "Coarse: GradNorm": sc.min() - ALPHA * recon_metrics["ppl"],
                    },
                    step=it + 1,
                )
                text_table.add_data(
                    "Coarse",
                    it + 1,
                    padded_s[sc.argmin()].split("<|endoftext|>")[1:-1],
                    sc.min(),
                    recon_metrics["ner"],
                    recon_metrics["ppl"],
                    recon_metrics["bleu"],
                    sc.min() - ALPHA * recon_metrics["ppl"],
                )

                print(
                    f"[Progress] Iter: {it}    Score: {sc.min()}   Sentence: {padded_s[sc.argmin()]}  Grad_norm: {sc.min() - ALPHA * recon_metrics['ppl']} {recon_metrics}",
                )

                if type(bestsent) == T.Tensor:
                    bestsent = bestsent.numpy()

                last_imp = it

                yield (
                    it,
                    movetype,
                    bestscore,
                    bgradloss,
                    bestsent,
                    bestmask,
                    recon_metrics,
                )

            if bestscore == 0.0:
                break
        else:
            recon_metrics = eval(
                ori_single,
                padded_s[sc.argmin()].split("<|endoftext|>")[1:-1],
                model,
                tokenizer,
            )
            print(
                f"[Stay] Iter: {it}    Score: {bestscore} Step best: {padded_s[sc.argmin()]}  Grad_norm: {sc.min() - ALPHA * recon_metrics['ppl']}  {recon_metrics}"
            )
            wandb.log(
                {
                    "Coarse: GradLoss": bestscore,
                    "Coarse: BLEU": recon_metrics["bleu"],
                    "Coarse: NER": recon_metrics["ner"],
                    "Coarse: PPL": recon_metrics["ppl"],
                    "Coarse: GradNorm": sc.min() - ALPHA * recon_metrics["ppl"],
                },
                step=it + 1,
            )
            text_table.add_data(
                "Coarse (stay)",
                it + 1,
                padded_s[sc.argmin()].split("<|endoftext|>")[1:-1],
                sc.min(),
                recon_metrics["ner"],
                recon_metrics["ppl"],
                recon_metrics["bleu"],
                sc.min() - ALPHA * recon_metrics["ppl"],
            )

        thespr = bestsc

        kopt = np.random.randint(3, 6)

        cutprobs = np.ones_like(bestwsc)

        cutprobs[~bestmask] = 0.0

        cutprobs[lfix] = 100
        cutprobs[-1 - rfix] = 100

        if it % 2 == 0 and len(bestsent) - lfix - rfix > 6:
            ncand = bestmask[lfix : len(bestsent) - rfix].sum()
            ncand = min(40, ncand)

            l, r = lfix, len(bestsent) - rfix
            candidates = np.random.choice(
                np.arange(l, r),
                replace=False,
                p=cutprobs[l:r] / cutprobs[l:r].sum(),
                size=(ncand,),
            )
            candidates.sort()

            movetype = f"GS {kopt}"

        else:

            ropt = np.random.randint(7, 15)

            try:
                start = np.random.randint(lfix + 1, len(bestsent) - ropt - rfix)

                l, r = start, start + ropt

                candidates = np.random.choice(
                    np.arange(l, r),
                    replace=False,
                    p=cutprobs[l:r] / cutprobs[l:r].sum(),
                    size=(min(ropt, (cutprobs[l:r] > 0).sum()),),
                )

            except:
                ropt = min(15, len(bestsent) - lfix - rfix - 2)
                start = np.random.randint(
                    lfix + 1, max(lfix + 2, len(bestsent) - ropt - rfix)
                )

                l, r = start, start + ropt
                candidates = np.random.choice(
                    np.arange(l, r),
                    replace=False,
                    p=cutprobs[l:r] / cutprobs[l:r].sum(),
                    size=(min(ropt, (cutprobs[l:r] > 0).sum()),),
                )

            candidates.sort()

            movetype = f"LS {kopt}"

        links = thespr[:, bestsent[candidates]][candidates]

        permsents = []
        permmasks = []

        i, v, o = shuffle_proposals(links, topk, bs, kopt)

        for j in range(bs):
            inds = [candidates[0]] + list(candidates[i[j]]) + [candidates[-1]]
            if v[j] > -1000:
                pieces = [bestsent[: inds[0]]]
                maskpieces = [bestmask[: inds[0]]]
                for k in range(kopt + 1):
                    pieces.append(bestsent[inds[o[j][k] - 1] : inds[o[j][k]]])
                    maskpieces.append(bestmask[inds[o[j][k] - 1] : inds[o[j][k]]])
                pieces.append(bestsent[inds[-1] :])
                newsent = np.concatenate(pieces, 0)

                maskpieces.append(bestmask[inds[-1] :])
                newmask = np.concatenate(maskpieces, 0)
            else:
                newsent, newmask = bestsent, bestmask

            permsents.append(newsent)
            permmasks.append(newmask)

        padded = T.stack(list(map(T.from_numpy, permsents)), 0).to(device)
        padded_s = tokenizer.batch_decode(padded)


def createModels():
    import GPUtil
    from transformers import AutoModelForCausalLM, GPT2Tokenizer

    tokenizer_GPT = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_GPT.pad_token = tokenizer_GPT.eos_token
    gpt2 = AutoModelForCausalLM.from_pretrained(
        "experiments/wikitext-103/wikitext-103-203456-samples/pretrained_models/batchsize_64/pretrain_epoch_31_iter_98000_linear_1e-05-randinit:False.model",
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    gpt2 = gpt2.to(device)
    gpt2.eval()

    return tokenizer_GPT, gpt2


def ibisAlgorithm(
    original_dy_dx, initial_sentence, model, tokenizer, attack_bs, original_sentence
):
    # no idea what all these args do
    b = 64
    B = 256
    patience = 128

    vocab = tokenizer.get_vocab()
    vocab = {vocab[i]: i for i in vocab}
    V = len(vocab)
    unbreakable = np.zeros((V,))
    for v in range(V):
        unbreakable[v] = vocab[v][0].lower() in "abcdefghijklmnopqrstuvwxyz"

    sentence = torch.LongTensor(tokenizer.encode(initial_sentence))
    before = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids[0]
    after = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids[0]

    mask = 1 - unbreakable[sentence]
    mask[0] = 1

    df = pd.DataFrame(columns=["round", "mode", "nll", "gradloss", "sentence"])
    for nch, k in enumerate(
        ibis(
            model,
            before,
            sentence,
            after,
            b,
            B,
            IBIS_MAX_STEP,
            patience,
            warminit=True,
            gluemask=mask,
            original_dy_dx=original_dy_dx,
            tokenizer=tokenizer,
            score_fn="grad",
            batchsize=attack_bs,
            original_sentence=original_sentence,
        )
    ):
        # if warminit is set to False, the ibis will start with random shuffles of the initialized sentence (that is to say, if we already have a good init, we should set warminit=True)
        if nch == 0:
            starting = k.item()
        else:
            df = df.append(
                {
                    "round": k[0],
                    "mode": k[1],
                    "nll": k[2],
                    "gradloss": k[3],
                    "sentence": tokenizer.decode(k[4][1:-1]),
                },
                ignore_index=True,
            )
            ibis_recon_metrics = k[6]
    return df[df["gradloss"] == df["gradloss"].min()], ibis_recon_metrics


def do_leakage(tokenizer, gpt2, original_sentence, ori_single, bs_recon, seed=1):
    import ast
    import re
    from functools import reduce

    np.random.seed(seed)
    torch.manual_seed(seed)

    bs_dy_dx = get_grad_gpt2(bs_recon, tokenizer, gpt2)
    ori_dy_dx = [get_grad_gpt2([o], tokenizer, gpt2) for o in original_sentence[:3]]
    ori_logs = [eval(ori_single, [o], gpt2, tokenizer) for o in original_sentence]
    bs_recon_metrics = eval(ori_single, bs_recon, gpt2, tokenizer)
    bs_strip = [
        "".join(re.split(r'( *[\.\?!][\'"\)\]]* *)', bs_recon[0])[:2])
    ]  # strip the first generated part

    bs_strip_dy_dx = get_grad_gpt2(bs_strip, tokenizer, gpt2)
    bs_strip_metrics = eval(ori_single, bs_strip, gpt2, tokenizer)

    # print(
    #     original_sentence,
    #     "GradNorm:",
    #     [torch.norm(odydx[LAYER_IDX]).cpu().numpy().item() for odydx in ori_dy_dx],
    #     "PPL:",
    #     [ologs["ppl"] for ologs in ori_logs],
    # )
    # print(
    #     bs_recon,
    #     "GradNorm:",
    #     torch.norm(bs_dy_dx[LAYER_IDX]).cpu().numpy().item(),
    #     "PPL:",
    #     bs_recon_metrics["ppl"],
    #     "BLEU:",
    #     bs_recon_metrics["bleu"],
    # )
    # print(
    #     bs_strip,
    #     "GradNorm:",
    #     torch.norm(bs_strip_dy_dx[LAYER_IDX]).cpu().numpy().item(),
    #     "PPL:",
    #     bs_strip_metrics["ppl"],
    #     "BLEU:",
    #     bs_strip_metrics["bleu"],
    # )

    if bs_strip_metrics["ppl"] < bs_recon_metrics["ppl"]:
        best_beamsearch = bs_strip
    else:
        best_beamsearch = bs_recon

    beamsearch_dy_dx = get_grad_gpt2(best_beamsearch, tokenizer, gpt2)
    if len(original_sentence) <= 64:
        original_dy_dx = get_grad_gpt2(original_sentence, tokenizer, gpt2)
        beamsearch_gradloss = (
            calculateGradLoss(original_dy_dx, beamsearch_dy_dx)["MSE"]
            .detach()
            .cpu()
            .numpy()
        ).item()
    else:
        original_dy_dx = None
        beamsearch_gradloss = 1e10

    wandb.log(
        {
            "Coarse: GradLoss": beamsearch_gradloss,
            "Coarse: BLEU": bs_recon_metrics["bleu"],
            "Coarse: NER": bs_recon_metrics["ner"],
            "Coarse: PPL": bs_recon_metrics["ppl"],
        },
        step=0,
    )
    text_table.add_data("Ori", 0, original_sentence, None, None, None, None, None)
    text_table.add_data(
        "Coarse",
        0,
        best_beamsearch,
        beamsearch_gradloss,
        bs_recon_metrics["ner"],
        bs_recon_metrics["ppl"],
        bs_recon_metrics["bleu"],
        None,
    )

    original_sentence_tokens = reduce(
        lambda x, y: x + y, [tokenizer.encode(s) for s in original_sentence]
    )
    best_beamsearch_tokens = reduce(
        lambda x, y: x + tokenizer.encode("<|endoftext|>") + y,
        [tokenizer.encode(s) for s in best_beamsearch],  # seperate sentences
    )

    best_ibis = best_beamsearch
    ibis_recon_metrics = bs_recon_metrics
    ibis_gradloss = beamsearch_gradloss

    best_local = best_beamsearch
    local_recon_metrics = bs_recon_metrics
    local_gradloss = beamsearch_gradloss

    #### IBIS ####
    if IBIS_MAX_STEP > 0:
        best_beamsearch_cat = tokenizer.decode(best_beamsearch_tokens)
        ibis_result, ibis_recon_metrics = ibisAlgorithm(
            original_dy_dx,
            best_beamsearch_cat,
            gpt2,
            tokenizer,
            attack_bs=len(best_beamsearch),
            original_sentence=original_sentence,
        )
        best_ibis = ibis_result["sentence"].item()
        best_ibis_sentences = best_ibis.split("<|endoftext|>")
        ibis_dy_dx = get_grad_gpt2(best_ibis_sentences, tokenizer, gpt2)
        if original_dy_dx is not None:
            ibis_gradloss = (
                calculateGradLoss(original_dy_dx, ibis_dy_dx)["MSE"]
                .detach()
                .cpu()
                .numpy()
            ).item()
        else:
            ibis_gradloss = 1e10

        # terminate early if the ibis algorithm perfectly recovers the original sentence
        if best_ibis == original_sentence:
            global LOCAL_MAX_STEP
            LOCAL_MAX_STEP = (
                0  # Do not run local swapping if global swapping gets perfect results
            )

        tokenized_ibis = tokenizer(best_ibis)["input_ids"]
        losses = [ibis_gradloss]
    else:
        tokenized_ibis = best_beamsearch_tokens
        losses = [beamsearch_gradloss]

    best_local = best_ibis
    local_recon_metrics = ibis_recon_metrics
    local_gradloss = ibis_gradloss

    ### LOCAL ###
    localswap = LocalSwap(gpt2, tokenizer)
    sep_idx = np.where(np.asarray(tokenized_ibis) == 50256)[0].tolist()
    # print(sep_idx)
    reorderGenerator = localswap.createAllOrderings(
        len(tokenized_ibis), without_order=True, sep_idx=sep_idx
    )

    current_tokens = tokenized_ibis
    all_tokens = original_sentence_tokens
    if args.insert:
        insertGenerator = localswap.createAllOrderings(len(all_tokens), sep_idx=sep_idx)
    else:
        insertGenerator = None

    last_imp = 0
    for i in range(1, LOCAL_MAX_STEP):
        reordered = localswap.reorderAlgorithm(
            original_dy_dx, current_tokens, reorderGenerator
        )
        inserted = localswap.insertAlgorithm(
            original_dy_dx, current_tokens, all_tokens, insertGenerator
        )

        best = min(
            [
                reordered,
                inserted,
                localswap.computeStats(current_tokens, original_dy_dx, do_split=True),
            ],
            key=lambda x: x["gradloss_mse"],
        )
        current_tokens = best["tokens"]
        if best["gradloss_mse"] != losses[-1]:
            reorderGenerator = localswap.createAllOrderings(
                len(current_tokens), without_order=True, sep_idx=sep_idx
            )
            if args.insert:
                insertGenerator = localswap.createAllOrderings(
                    len(all_tokens), sep_idx=sep_idx
                )
            else:
                insertGenerator = None

            sys.stdout.flush()
            losses.append(best["gradloss_mse"])

            local_recon_metrics = eval(ori_single, best["sentence"], gpt2, tokenizer)
            print(
                i,
                tokenizer.decode(current_tokens),
                "GradNorm:",
                best["gradloss_mse"] - ALPHA * local_recon_metrics["ppl"],
                "PPL:",
                local_recon_metrics["ppl"],
                "BLEU:",
                local_recon_metrics["bleu"],
            )
            wandb.log(
                {
                    "Fine: GradLoss": best["gradloss_mse"],
                    "Fine: BLEU": local_recon_metrics["bleu"],
                    "Fine: NER": local_recon_metrics["ner"],
                    "Fine: PPL": local_recon_metrics["ppl"],
                    "Fine: GradNorm": best["gradloss_mse"]
                    - ALPHA * local_recon_metrics["ppl"],
                },
                step=i + IBIS_MAX_STEP,
            )
            text_table.add_data(
                "Fine",
                i,
                best["sentence"],
                best["gradloss_mse"],
                local_recon_metrics["ner"],
                local_recon_metrics["ppl"],
                local_recon_metrics["bleu"],
                best["gradloss_mse"] - ALPHA * local_recon_metrics["ppl"],
            )
            last_imp = i
        else:
            wandb.log(
                {
                    "Fine: GradLoss": best["gradloss_mse"],
                    "Fine: BLEU": local_recon_metrics["bleu"],
                    "Fine: NER": local_recon_metrics["ner"],
                    "Fine: PPL": local_recon_metrics["ppl"],
                    "Fine: GradNorm": best["gradloss_mse"]
                    - ALPHA * local_recon_metrics["ppl"],
                },
                step=i + IBIS_MAX_STEP,
            )
            text_table.add_data(
                "Fine",
                i,
                best["sentence"],
                best["gradloss_mse"],
                local_recon_metrics["ner"],
                local_recon_metrics["ppl"],
                local_recon_metrics["bleu"],
                best["gradloss_mse"] - ALPHA * local_recon_metrics["ppl"],
            )

        if best["gradloss_mse"] == 0.0:
            print("Finished early due to perfect recovery!")
            break
        if i - last_imp > LOCAL_PATIENCE:
            break

    best_local = best["sentence"]
    local_gradloss = best["gradloss_mse"]

    res = {
        "Original": original_sentence,
        "After_BS": {
            "sent": bs_recon,
            "score": bs_recon_metrics,
            # "gradloss": beamsearch_gradloss,
        },
        "After_strip": {
            "sent": bs_strip,
            "score": bs_strip_metrics,
            # "gradloss": bs_strip_dy_dx,
        },
        "After_coarse": {
            "sent": best_ibis,
            "score": ibis_recon_metrics,
            # "gradloss": ibis_gradloss,
        },
        "After_fine": {
            "sent": best_local,
            "score": local_recon_metrics,
            # "gradloss": local_gradloss,
        },
    }
    return res


def parse_config(row, wandb):
    config = {}
    wandb.config.batch_size = row["batchsize"]
    wandb.config.batch_id = row["sample_num"]
    wandb.config.beam_size = row["sample_num"]
    wandb.config.iter = row["train_iteration"]
    wandb.config.scheduler = row["train_lrsched"]
    wandb.config.randinit = row["randinit"]
    wandb.config.freqknown = row["freqknown"]
    wandb.config.lr = row["train_lr"]
    return config


if __name__ == "__main__":
    IBIS_MAX_STEP = 100
    LOCAL_MAX_STEP = 100
    LOCAL_PATIENCE = 50

    LAYER_IDX = 68  # 0: wte; 1: wpe; 68: h.5.ln_2.weight

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--data", type=str, default="wikitext")
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--freqknown", action="store_true")
    parser.add_argument("--insert", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--n_beamheight", action="store_true")
    args = parser.parse_args()

    if args.data == "wikitext":
        if args.n_beamheight:
            args.fname = f"experiments/wikitext-103-lrsched/wikitext-103-203456-samples/bs_results_height_n_freqknown_{args.freqknown}.pkl"
        else:
            args.fname = f"experiments/wikitext-103-lrsched/wikitext-103-203456-samples/bs_results_height_1_freqknown_{args.freqknown}.pkl"
    else:
        if args.n_beamheight:
            args.fname = f"experiments/enron/enron-30000/bs_results_height_n_freqknown_{args.freqknown}.pkl"
        else:
            args.fname = f"experiments/enron/enron-30000/bs_results_height_1_freqknown_{args.freqknown}.pkl"
    ALPHA = args.alpha
    exp = pd.read_pickle(args.fname)

    run_name = f"bs={args.bs}"
    run = wandb.init(
        project="reorder-single",
        name=f"single-{run_name}-ibis-{IBIS_MAX_STEP}-local-{LOCAL_MAX_STEP}-layer-{LAYER_IDX}-insert{args.insert}-delete-{args.delete}",
    )
    text_table = wandb.Table(
        columns=[
            "Mode",
            "Iter",
            "Sentence",
            "GradLoss",
            "NER",
            "PPL",
            "BLEU",
            "GradNorm",
        ]
    )

    rows = exp[exp["batchsize"] == args.bs]

    parse_config(rows, wandb)

    tokenizer, gpt2 = createModels()
    if args.data == "wikitext":
        if args.n_beamheight:
            save_file = f"experiments/wikitext-103-lrsched/wikitext-103-203456-samples/reorder_single_beamheight_n/bs-{args.bs}/freqknown-{args.freqknown}-span-{IBIS_MAX_STEP}-token_{LOCAL_MAX_STEP}-alpha={args.alpha}_insert={args.insert}_delete={args.delete}.json"
        else:
            save_file = f"experiments/wikitext-103-lrsched/wikitext-103-203456-samples/reorder_single_beamheight_1/bs-{args.bs}/freqknown-{args.freqknown}-span-{IBIS_MAX_STEP}-token_{LOCAL_MAX_STEP}-alpha={args.alpha}_insert={args.insert}_delete={args.delete}.json"
    else:
        if args.n_beamheight:
            save_file = f"experiments/enron/enron-30000/reorder_single_beamheight_n/bs-{args.bs}/freqknown-{args.freqknown}-span-{IBIS_MAX_STEP}-token_{LOCAL_MAX_STEP}-alpha={args.alpha}_insert={args.insert}_delete={args.delete}.json"
        else:
            save_file = f"experiments/enron/enron-30000/reorder_single_beamheight_1/bs-{args.bs}/freqknown-{args.freqknown}-span-{IBIS_MAX_STEP}-token_{LOCAL_MAX_STEP}-alpha={args.alpha}_insert={args.insert}_delete={args.delete}.json"
    rows.to_pickle(
        f"experiments/wikitext-103-lrsched/wikitext-103-203456-samples/reorder_single_beamheight_1/bs-{args.bs}/bs_results_height_1_freqknown_False.pkl"
    )  # save orginal sentences

    res = {}
    for ir in range(len(rows)):
        ori = rows["original"].iloc[ir]
        ori_single = [rows["original single"].iloc[ir]]
        bs_single = rows["candidate single"].iloc[ir]

        if rows["bleu"].iloc[ir] == 1:
            res[ir] = {
                "Original": ori,
                "After_BS": {
                    "sent": ori,
                    "score": None,
                    # "gradloss": beamsearch_gradloss,
                },
                "After_strip": {
                    "sent": ori,
                    "score": None,
                    # "gradloss": bs_strip_dy_dx,
                },
                "After_coarse": {
                    "sent": ori,
                    "score": None,
                    # "gradloss": ibis_gradloss,
                },
                "After_fine": {
                    "sent": ori,
                    "score": None,
                    # "gradloss": local_gradloss,
                },
            }
        else:
            res[ir] = do_leakage(tokenizer, gpt2, ori, ori_single, [bs_single], 1)
        run.log({"reordering_samples": text_table})
        # save after each run
        with open(save_file, "w") as f:
            json.dump(res, f, indent=4, separators=(",", ": "))
