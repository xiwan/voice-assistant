//! Voice activity detection using Silero VAD v5 (ONNX).
//!
//! Model interface (16 kHz): input [1, 64+512] f32 normalized audio
//! (64 samples of context from the previous chunk + 512 new samples),
//! state [2,1,128] f32, sr int64 -> output [1,1] speech probability + new state.
//! Feeding 512 samples without the 64-sample context degrades accuracy badly.

use anyhow::Result;
use ort::session::Session;
use ort::value::Tensor;

pub const VAD_CHUNK: usize = 512; // 32 ms @ 16 kHz
const CONTEXT: usize = 64;
const STATE_LEN: usize = 2 * 1 * 128;

pub struct Vad {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl Vad {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            state: vec![0f32; STATE_LEN],
            context: vec![0f32; CONTEXT],
        })
    }

    pub fn reset(&mut self) {
        self.state = vec![0f32; STATE_LEN];
        self.context = vec![0f32; CONTEXT];
    }

    /// Probability of speech in a 512-sample chunk of 16 kHz mono i16 audio.
    pub fn predict(&mut self, chunk: &[i16]) -> Result<f32> {
        assert_eq!(chunk.len(), VAD_CHUNK);
        let samples: Vec<f32> = chunk.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut input = Vec::with_capacity(CONTEXT + VAD_CHUNK);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(&samples);
        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array(([1usize, CONTEXT + VAD_CHUNK], input))?,
            "state" => Tensor::from_array(([2usize, 1, 128], self.state.clone()))?,
            "sr" => Tensor::from_array(([1usize], vec![16000i64]))?,
        ]?)?;
        let (_, prob) = outputs[0].try_extract_raw_tensor::<f32>()?;
        let (_, new_state) = outputs[1].try_extract_raw_tensor::<f32>()?;
        self.state = new_state.to_vec();
        self.context = samples[VAD_CHUNK - CONTEXT..].to_vec();
        Ok(prob[0])
    }
}
