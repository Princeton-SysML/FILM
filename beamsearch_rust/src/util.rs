use std::path::PathBuf;

use rust_bert::{
    gpt2::{GPT2LMHeadModel, Gpt2Config},
    pipelines::{
        common::{ModelType, TokenizerOption},
        generation_utils::{Cache, GenerateConfig, LMHeadModel},
    },
    Config, RustBertError,
};
use tch::{nn, Device, Tensor};

pub(crate) struct GPT2ModelAndTokenizer {
    pub(crate) model: GPT2LMHeadModel,
    pub(crate) tokenizer: TokenizerOption,
    pub(crate) _var_store: nn::VarStore,
    pub(crate) device: Device,
}
pub(crate) fn create_model(
    device: Device,
    weights_path: Option<PathBuf>,
) -> Result<GPT2ModelAndTokenizer, RustBertError> {
    let generate_config: GenerateConfig = Default::default();
    let config_path = match weights_path {
        Some(ref x) => x.parent().unwrap().join("config.json"),
        None => generate_config.config_resource.get_local_path()?,
    };
    println!("Config path: {:#?}", config_path);
    let vocab_path = generate_config.vocab_resource.get_local_path()?;
    let merges_path = generate_config.merges_resource.get_local_path()?;
    let weights_path = match weights_path {
        Some(x) => x,
        None => generate_config.model_resource.get_local_path()?,
    };

    let mut _var_store = nn::VarStore::new(device);
    let tokenizer = TokenizerOption::from_file(
        ModelType::GPT2,
        vocab_path.to_str().unwrap(),
        Some(merges_path.to_str().unwrap()),
        false,
        None,
        None,
    )?;
    let config = Gpt2Config::from_file(config_path);
    let model = GPT2LMHeadModel::new(&_var_store.root(), &config);
    _var_store.load(weights_path)?;
    Ok(GPT2ModelAndTokenizer {
        model,
        tokenizer,
        _var_store,
        device,
    })
}

pub(crate) fn get_logits(gpt2: &GPT2ModelAndTokenizer, input_ids: &Tensor) -> Tensor {
    gpt2.model
        .forward_t(
            Some(input_ids),
            Cache::GPT2Cache(None),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap()
        .lm_logits
}
