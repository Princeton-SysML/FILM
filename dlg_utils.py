from collections import Counter
from pprint import pprint

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from torch.autograd import grad
from transformers import GPT2LMHeadModel


def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """print the `n` best samples according to the given `metric`"""
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(
                f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
            )
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")
        pprint(samples[i])


def calculateGradLoss(original_dy_dx, generated_s_dy_dx, loss_fns=["MSE", "cosine"]):
    grad_diff = {"MSE": 0, "cosine": 0}

    for gx, gy in zip(generated_s_dy_dx, original_dy_dx):
        if "MSE" in loss_fns:
            grad_diff["MSE"] += ((gx - gy) ** 2).sum()
        if "cosine" in loss_fns:
            gx = gx.reshape((-1))
            gy = gy.reshape((-1))
            grad_diff["cosine"] += 1 - torch.sum(gx * gy, dim=-1) / (
                torch.norm(gx, dim=-1) * torch.norm(gy, dim=-1) + 0.000001
            )

    return grad_diff


def calculatePerplexity(sentence, model, tokenizer):
    """exp(loss)"""
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ppl1 = torch.exp(loss)

    return ppl1


def get_grad_gpt2(sentence, tokenizer_GPT, gpt2):
    tokens = tokenizer_GPT(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=gpt2.config.max_length,
    )[
        "input_ids"
    ]  # TODO: change max_length

    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens).long()
    tokens = tokens.to(gpt2.device)
    input_shape = tokens.size()
    position_ids = torch.arange(
        0, input_shape[-1], dtype=torch.long, device=gpt2.device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    gpt2.zero_grad()
    gpt2_outputs = gpt2(
        input_ids=tokens, output_hidden_states=True, return_dict=True, labels=tokens,
    )
    new_dy_dx = grad(
        gpt2_outputs.loss, gpt2.transformer.parameters(), retain_graph=True
    )

    list_dy_dx = list((_.detach().clone() for _ in new_dy_dx))

    return list_dy_dx


def get_grad(bert, sentence, tokenizer, device, pass_token=False, batched=False):
    if pass_token:
        original = sentence
    else:
        if batched:
            original = tokenizer.batch_encode_plus(
                sentence, return_tensors="pt", padding=True, truncation=True
            )["input_ids"].to(device)
        else:
            if isinstance(sentence, str):
                original = tokenizer.encode(sentence, return_tensors="pt").to(device)
            else:
                original = tokenizer.encode(sentence[0], return_tensors="pt").to(device)
    input_shape = original.size()

    attention_mask = torch.ones(input_shape, device=device)
    head_mask = None
    head_mask = bert.get_head_mask(head_mask, bert.config.num_hidden_layers)

    output_attentions = bert.config.output_attentions
    output_hidden_states = bert.config.output_hidden_states
    return_dict = bert.config.use_return_dict

    bert.transformer.zero_grad()

    dlbrt_output = bert.transformer(
        x=bert.embeddings(original),
        attn_mask=attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = dlbrt_output[0]

    dy_dx = grad(hidden_states.sum(), bert.transformer.parameters())
    list_dy_dx = list((_.detach().clone() for _ in dy_dx))

    return list_dy_dx


def generate_sentences(
    LMhead: GPT2LMHeadModel,
    tokenizer,
    leak_results_words,
    rep=1,
    max_length=30,
    batch_size=1,
    max_rep=2,
    max_rep_dict=None,
    prompts_list=None,
    leak_results_tokens=None,
    device="cpu",
    temperature=1,
    mode="sample",
    top_p=0.95,
):
    sentences = []
    generated_token_lists = []
    num_parallel = 10

    stop_words = list(stopwords.words("english"))
    stop_words = " ".join(stop_words)
    stop_words_tokens = tokenizer.encode(stop_words, return_tensors="pt").to(device)

    if leak_results_tokens is None:
        leak_results_tokens = tokenizer.encode(
            leak_results_words, return_tensors="pt"
        ).to(device)

    token2word = {}
    max_rep_dict_ori = max_rep_dict
    if max_rep_dict_ori is None:
        max_rep_dict = {}
    for token_id in leak_results_tokens:
        token2word[token_id] = tokenizer.decode(int(token_id))
        if max_rep_dict_ori is None:
            if token_id in stop_words_tokens:
                max_rep_dict[token_id] = (
                    max_rep * 2
                )  ## Yangsibo: we can also customize this value
            else:
                max_rep_dict[token_id] = max_rep
    print(max_rep_dict)
    print(np.sum([v for (k, v) in max_rep_dict.items()]))
    print(token2word)

    def prefix_allowed_tokens_fn(batch_id, sent):
        rep_dict = Counter(sent.cpu().numpy().tolist())
        allowed = Counter(max_rep_dict)
        allowed.subtract(rep_dict)

        ret = [k for (k, v) in allowed.items() if v > 0]

        prompts_ids = [
            tokenizer.encode(prompt, return_tensors="pt").cpu()[0].numpy()[0]
            for prompt in prompts_list
        ]

        if len(ret) == 0:
            return (
                tokenizer.encode("###", return_tensors="pt").to(device).cpu()[0].numpy()
            )  ## TODO: should implement this as a stopping criteria

        # else:
        #     return ret
        # Yangsibo: it seems that this heuristic makes the generation worse
        if (
            len(sent) % max_length == 0
            or sent[-1]
            == tokenizer.encode(".", return_tensors="pt").to(device).cpu()[0].numpy()[0]
        ):  # Only allow prompt words at begining of sequences
            if len(set(prompts_ids) & set(ret)) > 0:
                return list(set(prompts_ids) & set(ret))
            else:
                return ret

        else:
            return list(set(ret) - set(prompts_ids))

    if prompts_list is None:
        prompts_list = leak_results_words  # Use each word in the inferred set as prompt

    for prompts in prompts_list:
        if prompts in ["[CLS]", "[SEP]", " "]:
            continue
        print(f'Using "{prompts}" as prompt ...')
        if mode == "sample":
            for r in range(rep):
                prompts_rep = [prompts for _ in range(num_parallel)]
                inputs = tokenizer(prompts_rep, return_tensors="pt", truncation=True)
                # print(prompts, inputs['input_ids'].shape)

                # Yangsibo: maybe we should tune the hyper-params here
                output_sequences = LMhead.generate(
                    input_ids=inputs["input_ids"].to(device),
                    max_length=max_length * batch_size,
                    min_length=max_length * batch_size,
                    do_sample=True,
                    top_k=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.2,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    output_scores=True,
                )

                end_sign = tokenizer.encode("###", return_tensors="pt").to(device)[0]
                # import pdb; pdb.set_trace()
                output_sequences = [
                    [token_id for token_id in output_sequence if token_id != end_sign]
                    for output_sequence in output_sequences
                ]
                if batch_size > 1:
                    output_sequences = [
                        shard_sequence(output_sequence, max_length)
                        for output_sequence in output_sequences
                    ]
                    # import pdb; pdb.set_trace()
                    generated_token = [
                        [
                            [tokenizer.decode(int(token_id)) for token_id in s]
                            for s in output_sequence
                        ]
                        for output_sequence in output_sequences
                    ]
                    texts = [
                        tokenizer.batch_decode(
                            output_sequence, skip_special_tokens=False
                        )
                        for output_sequence in output_sequences
                    ]
                else:
                    generated_token = [
                        [
                            tokenizer.decode(int(token_id))
                            for token_id in output_sequence
                        ]
                        for output_sequence in output_sequences
                    ]
                    texts = tokenizer.batch_decode(
                        output_sequences, skip_special_tokens=False
                    )
                print(texts)
                sentences.extend(texts)
                generated_token_lists.extend(generated_token)

        elif mode == "beam":

            num_beams = min(num_parallel * rep, len(leak_results_words))
            num_return = min(num_beams, 50)
            # import pdb; pdb.set_trace()
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True)
            output_sequences = LMhead.generate(
                input_ids=inputs["input_ids"].to(device),
                max_length=max_length * batch_size,
                min_length=max_length * batch_size,
                num_beams=num_beams,
                num_return_sequences=num_return,
                temperature=temperature,
                repetition_penalty=1.2,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            end_sign = tokenizer.encode("###", return_tensors="pt").to(device)[0]
            # import pdb; pdb.set_trace()
            output_sequences = [
                [token_id for token_id in output_sequence if token_id != end_sign]
                for output_sequence in output_sequences
            ]
            if batch_size > 1:
                output_sequences = [
                    shard_sequence(output_sequence, max_length)
                    for output_sequence in output_sequences
                ]
                # import pdb; pdb.set_trace()
                generated_token = [
                    [
                        [tokenizer.decode(int(token_id)) for token_id in s]
                        for s in output_sequence
                    ]
                    for output_sequence in output_sequences
                ]
                texts = [
                    tokenizer.batch_decode(output_sequence, skip_special_tokens=False)
                    for output_sequence in output_sequences
                ]
            else:
                generated_token = [
                    [tokenizer.decode(int(token_id)) for token_id in output_sequence]
                    for output_sequence in output_sequences
                ]
                texts = tokenizer.batch_decode(
                    output_sequences, skip_special_tokens=False
                )
            print(texts)
            sentences.extend(texts)
            generated_token_lists.extend(generated_token)

    return sentences, generated_token_lists


def shard_sequence(input, max_legnth):
    ori_len = len(input)
    ret = []
    for i in range(ori_len // max_legnth):
        s = input[i * max_legnth : (i + 1) * max_legnth]
        ret.append(s)
    return ret


def run_leakage(input, sequences, tokenizer, bert, optim, device):
    input_shape = input.size()
    inputs_embeds = bert.embeddings(input)
    attention_mask = torch.ones(input_shape, device=device)
    head_mask = None
    head_mask = bert.get_head_mask(head_mask, bert.config.num_hidden_layers)

    output_attentions = bert.config.output_attentions
    output_hidden_states = bert.config.output_hidden_states
    return_dict = bert.config.use_return_dict

    optim.zero_grad()
    dlbrt_output = bert.transformer(
        x=inputs_embeds,
        attn_mask=attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = dlbrt_output[0]
    hidden_states.sum().backward()  # can be replaced by any other loss functions

    leaked_seq_length = (
        bert.embeddings.position_embeddings.weight.grad.var(dim=-1).nonzero().max()
    )
    leaked_token_ids = (
        bert.embeddings.word_embeddings.weight.grad.var(dim=-1).nonzero().view(-1)
    )
    leaked_words = tokenizer.decode(leaked_token_ids)  # The order is not preserved.

    s1 = set(leaked_token_ids.tolist())
    s2 = set((k.item() for ids in input for k in ids))
    diff = s1 - s2
    leak_ratio = len(s1) / sum(len(s) for s in input)

    #     print("--" * 40)

    leaked_words = []
    for i in range(leaked_token_ids.size(0)):
        leaked_word = tokenizer.decode(leaked_token_ids[i])
        leaked_words.append(leaked_word)
    #     print("Leaked sentence length:", leaked_seq_length.item() + 1)
    # print("Leaked words:", "|".join(leaked_words))
    #     print("--" * 40)

    original_words = [tokenizer.decode(t) for seq in input for t in seq]
    #     print("Original sentence length:", len(original_words))
    # print("Original sentence:", "|".join(original_words))
    return {
        "leakratio": leak_ratio,
        "originalwords": original_words,
        "leakedwords": leaked_words,
        "leaktokenids": leaked_token_ids,
    }
