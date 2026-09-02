//! Wake word detection: Rust port of the openWakeWord streaming pipeline.
//!
//! Pipeline (all constants match openwakeword/utils.py `AudioFeatures`):
//!   1. melspectrogram.onnx: raw i16-as-f32 audio -> mel frames (32 bins, hop 160).
//!      Fed in 1280-sample steps with 480 samples of left context -> 8 new frames.
//!      Output transform: x / 10 + 2.
//!   2. embedding_model.onnx: last 76 mel frames [1,76,32,1] -> 96-dim embedding.
//!      One new embedding per 1280 samples (8 mel frames step).
//!   3. classifier (e.g. hey_jarvis.onnx): last 16 embeddings [1,16,96] -> score.

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::VecDeque;

const CHUNK: usize = 1280; // 80 ms @ 16 kHz
const MEL_CONTEXT: usize = 480; // 160 * 3 left-context samples
const MEL_BINS: usize = 32;
const MEL_WINDOW: usize = 76; // mel frames per embedding
const EMB_DIM: usize = 96;
const CLS_WINDOW: usize = 16; // embeddings per classification
const RAW_MAX: usize = 16000 * 10;
const MEL_MAX: usize = 970; // 10 * 97
const FEAT_MAX: usize = 120;

pub struct WakeWord {
    melspec: Session,
    embedding: Session,
    classifier: Session,
    mel_in: String,
    emb_in: String,
    cls_in: String,
    pending: Vec<i16>,
    raw: VecDeque<i16>,
    mel_buf: VecDeque<[f32; MEL_BINS]>,
    feats: VecDeque<[f32; EMB_DIM]>,
}

impl WakeWord {
    pub fn new(melspec_path: &str, embedding_path: &str, classifier_path: &str) -> Result<Self> {
        let load = |p: &str| -> Result<Session> {
            Session::builder()?
                .commit_from_file(p)
                .map_err(|e| anyhow!("failed to load {p}: {e}"))
        };
        let melspec = load(melspec_path)?;
        let embedding = load(embedding_path)?;
        let classifier = load(classifier_path)?;
        let mel_in = melspec.inputs[0].name.clone();
        let emb_in = embedding.inputs[0].name.clone();
        let cls_in = classifier.inputs[0].name.clone();
        let mut ww = Self {
            melspec,
            embedding,
            classifier,
            mel_in,
            emb_in,
            cls_in,
            pending: Vec::new(),
            raw: VecDeque::new(),
            mel_buf: VecDeque::new(),
            feats: VecDeque::new(),
        };
        ww.reset();
        Ok(ww)
    }

    /// Reset internal buffers (call after a detection to avoid re-triggering).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.raw.clear();
        self.mel_buf.clear();
        // openwakeword initializes the mel buffer with ones((76, 32))
        for _ in 0..MEL_WINDOW {
            self.mel_buf.push_back([1.0f32; MEL_BINS]);
        }
        self.feats.clear();
    }

    /// Feed 16 kHz mono i16 samples. Returns the max classifier score produced
    /// by this batch, or None if not enough audio has accumulated yet.
    pub fn feed(&mut self, samples: &[i16]) -> Result<Option<f32>> {
        self.pending.extend_from_slice(samples);
        let mut best: Option<f32> = None;
        while self.pending.len() >= CHUNK {
            let chunk: Vec<i16> = self.pending.drain(..CHUNK).collect();
            self.raw.extend(chunk);
            while self.raw.len() > RAW_MAX {
                self.raw.pop_front();
            }
            // melspec over the new chunk plus left context
            let n = (CHUNK + MEL_CONTEXT).min(self.raw.len());
            let start = self.raw.len() - n;
            let input: Vec<f32> = self.raw.iter().skip(start).map(|&s| s as f32).collect();
            for frame in self.run_melspec(input)? {
                self.mel_buf.push_back(frame);
            }
            while self.mel_buf.len() > MEL_MAX {
                self.mel_buf.pop_front();
            }
            if self.mel_buf.len() >= MEL_WINDOW {
                let emb = self.run_embedding()?;
                self.feats.push_back(emb);
                while self.feats.len() > FEAT_MAX {
                    self.feats.pop_front();
                }
            }
            if self.feats.len() >= CLS_WINDOW {
                let score = self.run_classifier()?;
                best = Some(best.map_or(score, |b: f32| b.max(score)));
            }
        }
        Ok(best)
    }

    fn run_melspec(&mut self, input: Vec<f32>) -> Result<Vec<[f32; MEL_BINS]>> {
        let n = input.len();
        let tensor = Tensor::from_array(([1usize, n], input))?;
        let outputs = self.melspec.run(ort::inputs![self.mel_in.as_str() => tensor]?)?;
        let (shape, data) = outputs[0].try_extract_raw_tensor::<f32>()?;
        let frames = *shape.iter().rev().nth(1).unwrap_or(&0) as usize;
        let mut result = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut row = [0f32; MEL_BINS];
            for (b, v) in row.iter_mut().enumerate() {
                // openwakeword melspec transform: x / 10 + 2
                *v = data[f * MEL_BINS + b] / 10.0 + 2.0;
            }
            result.push(row);
        }
        Ok(result)
    }

    fn run_embedding(&mut self) -> Result<[f32; EMB_DIM]> {
        let start = self.mel_buf.len() - MEL_WINDOW;
        let mut input = Vec::with_capacity(MEL_WINDOW * MEL_BINS);
        for frame in self.mel_buf.iter().skip(start) {
            input.extend_from_slice(frame);
        }
        let tensor = Tensor::from_array(([1usize, MEL_WINDOW, MEL_BINS, 1], input))?;
        let outputs = self.embedding.run(ort::inputs![self.emb_in.as_str() => tensor]?)?;
        let (_, data) = outputs[0].try_extract_raw_tensor::<f32>()?;
        let mut emb = [0f32; EMB_DIM];
        emb.copy_from_slice(&data[..EMB_DIM]);
        Ok(emb)
    }

    fn run_classifier(&mut self) -> Result<f32> {
        let start = self.feats.len() - CLS_WINDOW;
        let mut input = Vec::with_capacity(CLS_WINDOW * EMB_DIM);
        for feat in self.feats.iter().skip(start) {
            input.extend_from_slice(feat);
        }
        let tensor = Tensor::from_array(([1usize, CLS_WINDOW, EMB_DIM], input))?;
        let outputs = self.classifier.run(ort::inputs![self.cls_in.as_str() => tensor]?)?;
        let (_, data) = outputs[0].try_extract_raw_tensor::<f32>()?;
        Ok(data[0])
    }
}
