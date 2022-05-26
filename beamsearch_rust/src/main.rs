mod beamsearch;
mod util;

use beamsearch::BeamScoreReduction;
use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tch::Device;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct FutureCostProbs {
    bigram_probs: HashMap<i64, HashMap<i64, i64>>,
}

use clap::Parser;

use crate::beamsearch::{get_frequencies, order_words, BeamSearchOptions};
use crate::util::create_model;
#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    /// How a beam's perplexity should be reduced for scoring. Default is sum (i.e. add all perplexities in a beam together).
    #[clap(long, arg_enum, default_value_t = BeamScoreReduction::Sum)]
    beam_score_method: BeamScoreReduction,

    /// Path of input csv data
    #[clap(long)]
    datapath: String,

    /// Path to output csv data
    #[clap(long)]
    outputpath: String,

    /// Path to GPT2 model weights file. If not specified, uses GPT2 from huggingface.
    #[clap(long)]
    modelweights: Option<PathBuf>,

    /// Max length to assume during beamsearch
    #[clap(long, default_value_t = 40)]
    max_length: usize,

    /// Number of beams to use during search
    #[clap(long, default_value_t = 16)]
    num_beams: usize,

    /// Number of beams to use during search
    #[clap(long, default_value_t = 1)]
    beam_height: usize,

    /// Number of beam groups to use during search
    #[clap(long, default_value_t = 8)]
    num_beamgroups: usize,

    /// If specified, prevents ngrams of passed value from being repeated in a sentence.
    #[clap(long)]
    no_repeat_ngram_size: Option<usize>,

    /// Number of times to repeat with a penalty applied to repeat tokens
    #[clap(long, default_value_t = 1)]
    n_repetitions: usize,

    /// Diversity penalty to apply
    #[clap(long, default_value_t = 0.1)]
    diversity_penalty: f64,

    /// Sets token frequencies to be unknown during recoveries. Should not be set if frequencies are in the provided data file.
    #[clap(long)]
    no_frequencies: bool,

    /// The cuda device to use. If not specified, then use CPU
    #[clap(long)]
    cuda_device: Option<usize>,

    /// Specifies whether or not to use starting token frequencies when initializing beams for beam search. This may greatly increase the runtime and memory usage of the first iteration of search.
    #[clap(long)]
    use_start_tok_freqs: bool,
}

#[derive(Deserialize, Serialize, Debug)]
struct FrequencyData {
    word: String,
    ground_truth: i64,
    pred_norm_only: f64,
    pred_freq_only: f64,
    pred_norm_and_freq: f64,
}

#[derive(Deserialize, Serialize, Debug)]
struct InputData {
    original: Vec<String>,
    frequencies: Option<HashMap<i64, FrequencyData>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let input_file = File::open(args.datapath)?;
    let input_data: InputData = serde_json::from_reader(input_file)?;

    let device = match args.cuda_device {
        None => Device::Cpu,
        Some(x) => Device::Cuda(x),
    };

    let gpt2 = create_model(device, args.modelweights)?;

    let mut frequencies = get_frequencies(&gpt2, &input_data.original, args.max_length);

    if let Some(f) = input_data.frequencies {
        println!("Loading frequencies from saved data!");
        assert!(!args.no_frequencies);
        let base: f64 = f.values().map(|x| x.pred_freq_only).sum();
        let tokenized_inputs = gpt2.tokenizer.encode_list(
            &input_data.original,
            args.max_length,
            &TruncationStrategy::LongestFirst,
            1,
        );

        let max_token_len = tokenized_inputs
            .iter()
            .map(|v| v.token_ids.len())
            .max()
            .unwrap();
        let pred_to_freq = |pred: f64| -> i64 {
            let normalized = (pred / base) * (max_token_len * input_data.original.len()) as f64;
            match normalized < 1.5 {
                true => 2,                     // be overly-conservative and guess 2
                false => max_token_len as i64, // otherwise, we are on the tail-end of the dist. we guess high in this case.
            }
        };
        frequencies = frequencies
            .iter()
            .map(|(&k, _)| (k, pred_to_freq(f[&k].pred_norm_only)))
            .collect();
    }

    let beamsearch_opts = BeamSearchOptions {
        max_length: args.max_length,
        num_beams: args.num_beams,
        num_groups: args.num_beamgroups,
        diversity_penalty: args.diversity_penalty,
        max_device_beams: match device {
            Device::Cpu => 10000000, // We allow infinite concurrent beams on cpu, since memory is "infinite"
            Device::Cuda(_) => 256, // On cuda, we restrict number of concurrent beams to evaluate since GPU mem is limited
        },
        n_repetitions: args.n_repetitions,
        beam_height: args.beam_height,
        frequencies,
        use_start_tok_freqs: args.use_start_tok_freqs,
        freqknown: !args.no_frequencies,
        beam_perplexity_scoring: args.beam_score_method,
        no_repeat_ngram_size: args.no_repeat_ngram_size,
    };

    println!("Running for batch {:?}", input_data.original);
    let results = order_words(&gpt2, input_data.original, beamsearch_opts)?;

    let mut file = File::create(args.outputpath)?;
    write!(file, "{}", serde_json::to_string_pretty(&results)?)?;
    Ok(())
}
