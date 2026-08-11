// This module defines a tokenizer for Cohere models using byte-level Byte-Pair-Encoding.

// The tokenizers crate is available via cargo dependencies, providing Rust implementations
// of tokenization algorithms including BPE, WordPiece, and other models with decoders,
// normalizers, and pre-tokenizers out of the box.

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
        SwiGlu, SwiGluConfig,
    },
    tensor::{Device, Tensor, DType, Int, Backend},
};
use tokenizers::{
    Tokenizer as TkTokenizer,
    decoders::{self, Decoder},
    models::bpe::{BPE, BPEBuilder},
    normalizers::{NFC, NormalizerWrapper},
    pre_tokenizers::{Digits, ByteLevel, self, PreTokenizerWrapper},
};

pub struct Tokenizer {
    // The underlying tokenizers crate implementation
    tokenizer: TkTokenizer,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(None, None, false, 0.0)
    }
}

impl Tokenizer {
    pub fn new(
        vocab: Option<String>,
        merges: Option<String>, 
        use_default_system_prompt: bool,
        add_prefix_space: f32,
    ) -> Self {
        // Build vocabulary and merge files if provided, or use default
        let mut bpe_builder = BPEBuilder::default();
        
        // Set up tokenizer model based on inputs
        let bpe = if let (Some(vocab_path), Some(merges_path)) = (vocab.as_ref(), merges.as_ref()) {
            let bpe = bpe_builder
                .vocab(vocab_path)
                .merges(merges_path)
                .dropout(None)
                .continuing_subword_prefix("".to_string())
                .end_of_word_suffix("".to_string())
                .fuse_unk(false)
                .build()
                .expect("Failed to build BPE from files");
            bpe
        } else {
            let default_vocab = [
                ("\\PAD\\".to_string(), 0),
                ("\\UNK\\".to_string(), 1), 
                ("<CLS>".to_string(), 2),
                ("<SEP>".to_string(), 3),
                ("\\MASK\\".to_string(), 4),
                ("".to_string(), 5), // bos_token
            ];
            let bpe = bpe_builder
                .vocab_hashmap(default_vocab.iter().cloned().collect())
                .merges(vec![])
                .dropout(None)
                .continuing_subword_prefix("".to_string())
                .end_of_word_suffix("".to_string())
                .fuse_unk(false)
                .build()
                .expect("Failed to build BPE from default vocab");
            bpe
        };
        
        let mut tokenizer = TkTokenizer::new(bpe, None).expect("Failed to create Tokenizer");
        
        // Configure the tokenizer with specific normalizers and processors
        tokenizers::normalizers::NFC::default();
        tokenizer.set_normalizer(Some(tokenizers::normalizers::NFC::default().into()));
        
        let mut pre_tokenizers: Vec<PreTokenizerWrapper> = vec![
            Digits { individual_digits: true }.into(),
            ByteLevel::new(add_prefix_space, true).expect("Failed to create ByteLevel"),
        ];
        
        tokenizer.set_pre_tokenizer(Some(pre_tokenizers.remove(0)));
        let mut sequence = tokenizers::pre_tokenizers::sequence::Sequence::default();
        sequence.pre_tokenizers.push_back(pre_tokenizers.remove(0));
        pre_tokenizers.into_iter().for_each(|p| {
            sequence.pre_tokenizers.push_back(p);
        });
        tokenizer.set_pre_tokenizer(Some(sequence.into()));
        
        // Set up decoder
        let mut decoder = tokenizers::decoders::ByteLevel{add_prefix_space: true, ..Default::default()};
        if add_prefix_space != 0.0 {
            decoder.add_prefix_space = true;
        }
        tokenizer.set_decoder(Some(decoder));
        
        Self { tokenizer }
    }
}
