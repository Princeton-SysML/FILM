use std::{
    cmp::{self, min, Ordering},
    collections::{HashMap, HashSet},
    error::Error,
    ops::{Add, Mul},
};

use crate::util::{get_logits, GPT2ModelAndTokenizer};
use clap::ArgEnum;
use itertools::{izip, Itertools};
use rand::{prelude::SliceRandom, thread_rng};
use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::Serialize;
use tch::{Device, IndexOp, Kind, Tensor};

#[derive(Debug)]
pub(crate) struct Beam {
    pub(crate) state: Box<Tensor>,
    pub(crate) score: f64,
    pub(crate) perplexities: Vec<f64>,
    pub(crate) lengths: Vec<usize>,
    pub(crate) seen_ngrams: Option<HashMap<Vec<i64>, HashSet<i64>>>,
}

impl Clone for Beam {
    fn clone(&self) -> Self {
        Self {
            state: Box::new(self.state.shallow_clone()),
            score: self.score,
            lengths: self.lengths.clone(),
            perplexities: self.perplexities.clone(),
            seen_ngrams: self.seen_ngrams.clone(),
        }
    }
}

impl Beam {
    pub(crate) fn advance(
        &self,
        sent_idx: usize,
        token: i64,
        score: f64,
        perplexity: f64,
        no_repeat_ngram: Option<Vec<i64>>,
    ) -> Beam {
        let mut state = self.state.copy();
        let _ = state
            .i(sent_idx as i64)
            .i(self.lengths[sent_idx] as i64)
            .fill_(token);

        let mut lengths = self.lengths.clone();
        lengths[sent_idx] += 1;

        let mut new_perplexities = self.perplexities.clone();
        new_perplexities[sent_idx] += perplexity;

        let mut new_ngrams = self.seen_ngrams.as_ref().cloned();

        if let Some(n) = no_repeat_ngram {
            new_ngrams
                .as_mut()
                .unwrap()
                .entry(n)
                .or_default()
                .insert(token);
        }

        Beam {
            state: Box::new(state),
            score,
            lengths,
            perplexities: new_perplexities,
            seen_ngrams: new_ngrams,
        }
    }

    pub(crate) fn get_sent_last_n_toks(&self, sent_idx: usize, n_toks: usize) -> Option<Vec<i64>> {
        let length = self.lengths[sent_idx];
        if length >= n_toks {
            let suffix = self
                .state
                .i(sent_idx as i64)
                .slice(0, (length - n_toks) as i64, length as i64, 1)
                .data()
                .into();
            return Some(suffix);
        }
        return None;
    }

    pub(crate) fn get_sent_vec(&self, idx: usize) -> Vec<i64> {
        self.state.i(idx as i64).data().into()
    }

    pub(crate) fn convert_to_string(&self, model: &GPT2ModelAndTokenizer) -> Vec<String> {
        (0..self.lengths.len())
            .map(|idx| self.get_sent_vec(idx))
            .map(|v| model.tokenizer.decode(&v, true, true))
            .collect()
    }

    fn get_remaining_token_counts(&self, token_counts: &HashMap<i64, i64>) -> HashMap<i64, i64> {
        let mut allowed_toks = token_counts.clone();
        for idx in 0..self.lengths.len() {
            let vector = self.get_sent_vec(idx);
            for val in vector {
                *allowed_toks.entry(val).or_insert(0) -= 1;
            }
        }

        allowed_toks
            .iter()
            .filter(|(&_k, &v)| v > 0)
            .map(|(&k, &v)| (k, v))
            .collect()
    }
    // Returns remaining tokens for a beam search state
    fn get_allowed_tokens(&self, token_counts: &HashMap<i64, i64>) -> HashSet<i64> {
        let token_counts = self.get_remaining_token_counts(token_counts);
        let set: HashSet<i64> = HashSet::from_iter(
            token_counts
                .iter()
                .filter(|(_k, &v)| v > 0)
                .map(|(&k, _v)| k),
        );
        set
    }

    fn get_sum_ppl(&self) -> f64 {
        self.perplexities.iter().sum::<f64>()
    }
}

impl PartialOrd for Beam {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl PartialEq for Beam {
    fn eq(&self, other: &Self) -> bool {
        (0..self.lengths.len())
            .map(|idx| self.get_sent_vec(idx) == other.get_sent_vec(idx))
            .all(|x| x)
    }
}

impl Eq for Beam {}

impl Ord for Beam {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BeamGroup {
    beams: Vec<Beam>,
}

/// Computes Hamming diversity between two beam groups.
/// Hamming diversity for two beams A,B is given by minimum number of token substitutions needed to convert from A to B.
/// Each element in the returned tensor is the sum of Hamming diversity for the beam

#[derive(Clone)]
pub(crate) struct BeamSearchOptions {
    pub(crate) max_length: usize,
    pub(crate) num_beams: usize,
    pub(crate) num_groups: usize,
    pub(crate) diversity_penalty: f64,
    pub(crate) max_device_beams: usize,
    pub(crate) frequencies: HashMap<i64, i64>,
    pub(crate) freqknown: bool,
    pub(crate) use_start_tok_freqs: bool,
    pub(crate) beam_height: usize,
    pub(crate) n_repetitions: usize,
    pub(crate) beam_perplexity_scoring: BeamScoreReduction,
    pub(crate) no_repeat_ngram_size: Option<usize>,
}

#[derive(Serialize, Debug)]
struct SingleBeamResult {
    result: Vec<String>,
    perplexity: f64,
    beam_score: f64,
}

#[derive(Serialize, Debug)]
pub(crate) struct BeamSearchResults {
    original: Vec<String>,
    results: Vec<SingleBeamResult>,
}

pub(crate) fn get_frequencies(
    gpt2: &GPT2ModelAndTokenizer,
    sentences: &[String],
    max_length: usize,
) -> std::collections::HashMap<i64, i64> {
    let tokenized_inputs =
        gpt2.tokenizer
            .encode_list(sentences, max_length, &TruncationStrategy::LongestFirst, 1);

    let mut token_counts: HashMap<i64, i64> = HashMap::new();

    for data in tokenized_inputs {
        for id in data.token_ids {
            *token_counts.entry(id).or_insert(0) += 1;
        }
    }
    token_counts
}

#[derive(Debug, Clone, Copy, ArgEnum)]
pub enum BeamScoreReduction {
    Sum,
    Min,
    MinAvg,
}

fn reduce_beam_perplexity(scoring_method: &BeamScoreReduction, beam: &Beam) -> f64 {
    match scoring_method {
        BeamScoreReduction::Sum => beam.perplexities.iter().sum(),
        BeamScoreReduction::Min => beam.perplexities.iter().cloned().reduce(f64::min).unwrap(),
        BeamScoreReduction::MinAvg => izip!(
            beam.lengths.iter().cloned(),
            beam.perplexities.iter().cloned()
        )
        .map(|(l, p)| p / l as f64)
        .reduce(f64::min)
        .unwrap(),
    }
}

pub(crate) fn order_words(
    gpt2: &GPT2ModelAndTokenizer,
    sentences: Vec<String>,
    options: BeamSearchOptions,
) -> Result<BeamSearchResults, Box<dyn Error>> {
    let batch_size = sentences.len();

    let tokenized_inputs = gpt2.tokenizer.encode_list(
        &sentences,
        options.max_length,
        &TruncationStrategy::LongestFirst,
        1,
    );

    let max_token_len = tokenized_inputs
        .iter()
        .map(|v| v.token_ids.len())
        .max()
        .unwrap();
    // let max_token_len = options.max_length;
    println!("Using max sentence length {max_token_len}!");

    let input_ids = Tensor::ones(
        &[sentences.len() as i64, (max_token_len + 1) as i64],
        (Kind::Int64, Device::Cpu),
    )
    .fill_(50256);

    // construct input tensor + count tokens
    for (sent_idx, data) in tokenized_inputs.iter().enumerate() {
        for (token_idx, id) in data.token_ids.iter().enumerate() {
            let _ = input_ids
                .get(sent_idx.try_into()?)
                .get(token_idx.try_into()?)
                .fill_(*id);
        }
    }

    // filter out possible start tokens
    let mut start_tokens: Vec<i64> = vec![];
    for (&k, &count) in &options.frequencies {
        let decoded = gpt2.tokenizer.decode(&[k], true, false);
        if decoded.chars().next().unwrap().is_uppercase() {
            let toks = match options.use_start_tok_freqs {
                true => vec![k].repeat(count as usize),
                false => vec![k],
            };
            start_tokens.extend(toks);
        }
    }

    if start_tokens.len() < batch_size {
        let mult = batch_size as f64 / start_tokens.len() as f64;
        let mult = mult.ceil() as usize;
        println!("Extending starting tokens list with multiplicity {mult}");
        let tmp = start_tokens.clone();
        let iter = tmp.iter().cycle().take(start_tokens.len() * mult);
        start_tokens.extend(iter);
    }

    println!("Found {} possible starting tokens!", start_tokens.len());

    let mut beam_results: Vec<Beam> = Vec::new();
    for _iter in 0..options.n_repetitions {
        let mut beams = beamsearch(gpt2, &start_tokens, &options, &beam_results);
        beam_results.append(&mut beams);
    }

    let mut results: Vec<SingleBeamResult> = beam_results
        .drain(..)
        .map(|b| SingleBeamResult {
            result: b.convert_to_string(gpt2),
            perplexity: reduce_beam_perplexity(&options.beam_perplexity_scoring.clone(), &b),
            beam_score: b.score,
        })
        .collect();
    results.sort_unstable_by(|a, b| b.perplexity.partial_cmp(&a.perplexity).unwrap());

    Ok(BeamSearchResults {
        original: sentences,
        results,
    })
}

fn beamsearch(
    gpt2: &GPT2ModelAndTokenizer,
    initial_beam_states: &[i64],
    options: &BeamSearchOptions,
    previous_results: &Vec<Beam>,
) -> Vec<Beam> {
    tch::Cuda::cudnn_set_benchmark(true);
    let _guard = tch::no_grad_guard();

    let max_sentence_length = min(
        options.frequencies.values().sum(),
        options.max_length as i64,
    );
    let max_iterations: i64 = min(
        options.frequencies.values().sum(),
        (options.max_length * options.beam_height) as i64,
    ) - (options.beam_height as i64);

    let mut init_beam_states = initial_beam_states.to_owned();
    init_beam_states.shuffle(&mut thread_rng());
    let initial_beam_states: Vec<Vec<i64>> = init_beam_states
        .chunks_exact(options.beam_height)
        .map(|x| x.to_vec())
        .collect();

    let initial_beam_tensors: Vec<Tensor> = initial_beam_states
        .iter()
        .map(|v| {
            Tensor::stack(
                &v.iter()
                    .map(|&x| {
                        Tensor::of_slice(
                            &[vec![x], vec![50256; max_sentence_length as usize]].concat(),
                        )
                    })
                    .collect::<Vec<Tensor>>(),
                0,
            )
        })
        .collect();

    let initial_beams: Vec<Beam> = initial_beam_tensors
        .iter()
        .map(|t| Beam {
            state: Box::new(t.shallow_clone()),
            score: 0.0,
            perplexities: vec![0.0; options.beam_height],
            lengths: vec![1; options.beam_height],
            seen_ngrams: options.no_repeat_ngram_size.and(Some(HashMap::new())),
        })
        .collect();

    let mut beam_groups: Vec<BeamGroup> = vec![BeamGroup {
        beams: initial_beams,
    }];

    beam_groups.extend(
        beam_groups
            .iter()
            .cycle()
            .take(options.num_groups - beam_groups.len())
            .cloned()
            .collect::<Vec<BeamGroup>>(),
    );

    let mut beam_s_time = time::Instant::now();

    // Penalize previously seen n-grams
    let previously_seen_ngrams = previous_results
        .iter()
        .map(|b| b.seen_ngrams.as_ref().unwrap())
        .cloned()
        .reduce(|mut a, b| {
            b.iter().for_each(|(k, v)| {
                a.entry(k.to_owned()).or_default().extend(v.iter());
            });
            a
        })
        .unwrap_or_default();

    for iteration in 1..=max_iterations {
        let new_s_time = time::Instant::now();
        let remaining_tokens: i64 = beam_groups[0].beams[0]
            .get_remaining_token_counts(&options.frequencies)
            .values()
            .sum();
        println!(
            "iteration {}/{} {:?} (remaining: {})",
            iteration,
            max_iterations,
            new_s_time - beam_s_time,
            remaining_tokens,
        );

        beam_s_time = time::Instant::now();

        debug_assert!(
            beam_groups.iter().all(|b| !b.beams.is_empty()),
            "Fail assert!"
        );
        let grouped_scores: Vec<Tensor> = {
            // Compute scores in chunks which will fit on the GPU
            let log_softmax_scores = {
                debug_assert!(!beam_groups.is_empty(), "No beam groups!");
                debug_assert!(
                    !beam_groups[0].beams.is_empty(),
                    "No beams in first beamgroup!"
                );
                // "flatten" all groups into single-sentence tensors
                let beam_tensors: Vec<(usize, Tensor)> = beam_groups
                    .iter()
                    .flat_map(|g| {
                        g.beams.iter().flat_map(|b| {
                            (0..b.lengths.len())
                                .map(|idx| (b.lengths[idx], b.state.i(idx as i64).shallow_clone()))
                        })
                    })
                    .collect();

                let all_tensors: Vec<Tensor> = beam_tensors
                    .chunks(options.max_device_beams as usize)
                    .into_iter()
                    .flat_map(|vals| {
                        let tensors: Vec<Tensor> =
                            vals.iter().map(|(_l, t)| t.shallow_clone()).collect();
                        let tensor_slice = Tensor::stack(&tensors, 0).to(gpt2.device);
                        let logits = get_logits(gpt2, &tensor_slice);
                        let logits_log_softmax = logits.log_softmax(-1, Kind::Float);
                        // get only relevant logits
                        vals.iter().enumerate().map(move |(idx, (l, _t))| {
                            // gpt2 output at index l - 1 is prediction for slot l in sentence
                            let l = (*l as i64) - 1;
                            logits_log_softmax.i(idx as i64).i(l).to(Device::Cpu)
                        })
                    })
                    .collect();
                Tensor::stack(&all_tensors, 0)
            };

            // split back up into individual beams
            let beam_split_scores = log_softmax_scores.split((options.beam_height) as i64, 0);

            // re-group beam scores
            let mut tmp: Vec<Tensor> = vec![];
            let mut curr_idx = 0;
            for group in &beam_groups {
                let beam_split_tmp = &beam_split_scores[curr_idx..(curr_idx + group.beams.len())];
                debug_assert!(
                    !beam_split_tmp.is_empty(),
                    "FAIL STACK: {curr_idx} {beam_groups:#?}"
                );
                let group_scores = Tensor::stack(beam_split_tmp, 0);
                tmp.push(group_scores);
                curr_idx += group.beams.len();
            }

            tmp
        };

        let mut new_groups: Vec<BeamGroup> = vec![];
        for (idx, (group, scores)) in izip!(beam_groups.drain(..), grouped_scores).enumerate() {
            // mask out disallowed tokens
            let allowed_tokens_mask = Tensor::zeros_like(&scores)
                .fill_(f64::NEG_INFINITY)
                .to(gpt2.device);

            let filled_sentences_mask = Tensor::zeros_like(&scores).to(gpt2.device);
            let penalize_ngrams_mask = Tensor::zeros_like(&scores).to(gpt2.device);
            for (beam_idx, beam) in group.beams.iter().enumerate() {
                // Remove penalty from allowed tokens
                let allowed_toks = match options.freqknown {
                    true => beam.get_allowed_tokens(&options.frequencies),
                    false => options.frequencies.keys().into_iter().copied().collect(),
                };

                let allowed_toks_tensor =
                    Tensor::of_slice(&allowed_toks.iter().copied().collect::<Vec<i64>>())
                        .to(gpt2.device);

                let _ = allowed_tokens_mask.i(beam_idx as i64).index_fill_(
                    -1,
                    &allowed_toks_tensor,
                    0.0,
                );

                // mask out sentences which are already full
                for (sent_idx, sent_len) in beam.lengths.iter().enumerate() {
                    if sent_len >= &(max_sentence_length as usize) {
                        let _ = filled_sentences_mask
                            .i(beam_idx as i64)
                            .i(sent_idx as i64)
                            .fill_(f64::NEG_INFINITY);
                    }
                }

                // penalize repeated n-grams (https://github.com/huggingface/transformers/blob/c85547af2b69f9082bcd7bac97092b1d162f3fdc/src/transformers/generation_logits_process.py#L327)
                if let Some(n) = options.no_repeat_ngram_size {
                    for sent_idx in 0..beam.lengths.len() {
                        let mut banned_tokens: Vec<i64> = vec![];
                        let suffix = beam.get_sent_last_n_toks(sent_idx, n - 1);
                        if let Some(suffix) = suffix {
                            let entry = previously_seen_ngrams.get(&suffix);
                            if let Some(ngrams) = entry {
                                banned_tokens.extend(ngrams.iter());
                            }
                        }

                        let banned_tokens_tensor = Tensor::of_slice(&banned_tokens).to(gpt2.device);
                        let _ = penalize_ngrams_mask
                            .i(beam_idx as i64)
                            .i(sent_idx as i64)
                            .index_fill_(-1, &banned_tokens_tensor, -10E10);
                    }
                    for sent_idx in 0..beam.lengths.len() {
                        let mut banned_tokens: Vec<i64> = vec![];
                        let suffix = beam.get_sent_last_n_toks(sent_idx, n - 1);
                        if let Some(suffix) = suffix {
                            let entry = beam.seen_ngrams.as_ref().and_then(|g| g.get(&suffix));
                            if let Some(ngrams) = entry {
                                banned_tokens.extend(ngrams.iter());
                            }
                        }

                        let banned_tokens_tensor = Tensor::of_slice(&banned_tokens).to(gpt2.device);
                        let _ = penalize_ngrams_mask
                            .i(beam_idx as i64)
                            .i(sent_idx as i64)
                            .index_fill_(-1, &banned_tokens_tensor, -10E10);
                    }
                }
            }

            let mut diversity_score = scores
                .copy()
                .to(gpt2.device)
                .add(allowed_tokens_mask)
                .add(filled_sentences_mask)
                .add(penalize_ngrams_mask);

            if idx > 0 {
                let beam_states = Tensor::stack(
                    &new_groups[..idx]
                        .iter()
                        .flat_map(|g| g.beams.iter().map(|b| b.state.shallow_clone()))
                        .collect::<Vec<Tensor>>(),
                    0,
                )
                .flatten(0, -1)
                .to(gpt2.device);
                let score_penalty = beam_states
                    .bincount::<Tensor>(None, 50257)
                    .mul(-1.0 * options.diversity_penalty);
                diversity_score += score_penalty;
            }

            let mut final_score = diversity_score.copy();
            let mut ppl_scores = Tensor::zeros_like(&final_score);
            for (beam_idx, beam) in group.beams.iter().enumerate() {
                let _ = ppl_scores.i(beam_idx as i64).fill_(reduce_beam_perplexity(
                    &options.beam_perplexity_scoring,
                    beam,
                ));
            }
            final_score += ppl_scores;

            let (topk_values, topk_tokens) =
                final_score
                    .flatten(0, -1)
                    .topk(options.num_beams as i64, -1, true, true);
            let (topk_values, topk_tokens) =
                (topk_values.to(Device::Cpu), topk_tokens.to(Device::Cpu));

            let mut new_beams: Vec<Beam> = vec![];
            for idx in 0..topk_values.size1().unwrap() {
                let (topk_value, topk_token) = (
                    topk_values.double_value(&[idx as i64]),
                    topk_tokens.int64_value(&[idx as i64]),
                );

                // extract beam and sentence index from topk_idx
                let beam_idx = topk_token / (options.beam_height * 50257) as i64;
                let sent_idx = (topk_token / 50257) % options.beam_height as i64;
                if group.beams[beam_idx as usize].lengths[sent_idx as usize]
                    >= max_sentence_length as usize
                {
                    continue;
                }
                let token = topk_token % 50257;

                let new_beam = group.beams[beam_idx as usize].advance(
                    sent_idx as usize,
                    token,
                    topk_value,
                    scores.i(beam_idx).i(sent_idx).double_value(&[token]),
                    options.no_repeat_ngram_size.and_then(|n| {
                        group.beams[beam_idx as usize]
                            .get_sent_last_n_toks(sent_idx as usize, n - 1)
                    }),
                );

                new_beams.push(new_beam);
            }

            debug_assert!(!new_beams.is_empty(), "No new beams!");
            let mut seen_beams: HashSet<Vec<Vec<i64>>> = HashSet::new();

            let mut unique_beams: Vec<Beam> = vec![];
            for b in new_beams.drain(..) {
                let vec_beam = b.state.data().into();
                if !seen_beams.contains(&vec_beam) {
                    seen_beams.insert(vec_beam);
                    unique_beams.push(b);
                }
            }

            new_groups.push({
                BeamGroup {
                    beams: unique_beams,
                }
            })
        }
        beam_groups = new_groups;
    }

    let top_group_beams: Vec<Beam> = beam_groups
        .iter()
        .map(|g| {
            g.beams
                .iter()
                .max_by(|x, y| x.get_sum_ppl().partial_cmp(&y.get_sum_ppl()).unwrap())
                .unwrap()
        })
        .cloned()
        .collect();
    top_group_beams
        .iter()
        .sorted_by(|a, b| b.get_sum_ppl().partial_cmp(&a.get_sum_ppl()).unwrap())
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, error::Error};

    use rust_tokenizers::tokenizer::TruncationStrategy;
    use tch::Device;

    use crate::{beamsearch::get_frequencies, util::create_model};

    use super::{beamsearch, order_words, BeamSearchOptions};

    #[test]
    fn test_beamsearch() -> Result<(), Box<dyn Error>> {
        let gpt2 = create_model(Device::Cpu, None)?;
        let sent = "The dog is very fat.";
        let tokens = &gpt2
            .tokenizer
            .encode_list(&[sent], 40, &TruncationStrategy::LongestFirst, 1)[0];
        let mut freqs: HashMap<i64, i64> = HashMap::new();
        for t in &tokens.token_ids {
            *freqs.entry(*t).or_insert(0) += 1;
        }
        let options = BeamSearchOptions {
            max_length: 40,
            num_beams: 16,
            num_groups: 8,
            diversity_penalty: 1.0,
            max_device_beams: 128,
            frequencies: freqs,
            freqknown: true,
            use_start_tok_freqs: false,
            beam_height: 1,
            beam_perplexity_scoring: super::BeamScoreReduction::Sum,
            no_repeat_ngram_size: None,
            n_repetitions: 2,
        };
        let results = beamsearch(&gpt2, &vec![tokens.token_ids[0]], &options, &Vec::new());
        for result in results {
            println!("{:#?}", result);
        }

        Ok(())
    }

    #[test]
    fn test_order_words() -> Result<(), Box<dyn Error>> {
        let gpt2 = create_model(Device::Cuda(1), None)?;
        let sentences =            [
        "Guitar Hero Encore: Rocks the80sfor the PlayStation 2, which was released in July 2007, was the final game developed by Harmonix for the series.".to_string(),
        "The increase in scale was a challenge to the director, who had been accustomed to small casts and limited locations.".to_string(),
        "By distancing himself from the work, he believed he provided the band with a fresh perspective on their material each time he rejoined them.".to_string(),
        "Mammoths born with at least one copy of the dominant allele would have had dark coats, while those with two copies of the recessive allele would have had light coats.".to_string()
    ];
        let freqs = get_frequencies(&gpt2, &sentences, 40);
        let options = BeamSearchOptions {
            max_length: 40,
            num_beams: 32,
            num_groups: 2,
            diversity_penalty: 1.0,
            max_device_beams: 256,
            frequencies: freqs,
            freqknown: true,
            use_start_tok_freqs: false,
            beam_height: 1,
            beam_perplexity_scoring: super::BeamScoreReduction::Sum,
            no_repeat_ngram_size: Some(2),
            n_repetitions: 2,
        };
        let result = order_words(&gpt2, sentences.into(), options)?;
        dbg!(result);

        Ok(())
    }
    #[test]
    fn test_order_words_2() -> Result<(), Box<dyn Error>> {
        let gpt2 = create_model(Device::Cuda(1), None)?;
        let sentences = ["My dog is cute.".to_string(), "The cat is fat.".to_string()];
        let freqs = get_frequencies(&gpt2, &sentences, 40);
        let options = BeamSearchOptions {
            max_length: 40,
            num_beams: 32,
            num_groups: 2,
            diversity_penalty: 1.0,
            max_device_beams: 256,
            frequencies: freqs,
            freqknown: true,
            use_start_tok_freqs: false,
            beam_height: 2,
            beam_perplexity_scoring: super::BeamScoreReduction::Sum,
            no_repeat_ngram_size: Some(2),
            n_repetitions: 2,
        };
        let result = order_words(&gpt2, sentences.into(), options)?;
        dbg!(result);

        Ok(())
    }
}
