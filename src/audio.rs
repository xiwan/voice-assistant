//! Microphone capture via cpal, downmixed to mono and resampled to 16 kHz i16.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;

pub const TARGET_RATE: u32 = 16000;

/// Holds the live input stream. Must be kept alive for capture to continue.
pub struct Capture {
    _stream: cpal::Stream,
}

impl Capture {
    /// Start capturing from the default input device.
    /// Sends chunks of 16 kHz mono i16 samples to `tx`.
    pub fn start(tx: Sender<Vec<i16>>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device (check mic permission)"))?;
        eprintln!("[audio] device: {}", device.name().unwrap_or_default());
        let config = device.default_input_config()?;
        let src_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        eprintln!(
            "[audio] {} Hz, {} ch, {:?} -> 16000 Hz mono",
            src_rate,
            channels,
            config.sample_format()
        );

        let err_fn = |e| eprintln!("[audio] stream error: {e}");
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let mut rs = Resampler::new(src_rate, TARGET_RATE);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &_| {
                        let mono: Vec<f32> = data
                            .chunks(channels)
                            .map(|f| f.iter().sum::<f32>() / channels as f32)
                            .collect();
                        let out = rs.process(&mono);
                        if !out.is_empty() {
                            // Never block the audio callback thread: drop the
                            // chunk if the consumer is busy (e.g. agent running).
                            let _ = tx.try_send(out);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let mut rs = Resampler::new(src_rate, TARGET_RATE);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[i16], _: &_| {
                        let mono: Vec<f32> = data
                            .chunks(channels)
                            .map(|f| {
                                f.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                    / channels as f32
                            })
                            .collect();
                        let out = rs.process(&mono);
                        if !out.is_empty() {
                            let _ = tx.try_send(out);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            fmt => return Err(anyhow!("unsupported sample format {fmt:?}")),
        };
        stream.play()?;
        Ok(Self { _stream: stream })
    }
}

/// List input devices (for `devices` subcommand).
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();
    for dev in host.input_devices()? {
        let name = dev.name().unwrap_or_default();
        let marker = if name == default_name { " (default)" } else { "" };
        println!("{name}{marker}");
    }
    Ok(())
}

/// Streaming linear-interpolation resampler (f32 in, i16 out).
struct Resampler {
    ratio: f64,
    pos: f64,
    last: f32,
}

impl Resampler {
    fn new(src: u32, dst: u32) -> Self {
        Self {
            ratio: src as f64 / dst as f64,
            pos: 0.0,
            last: 0.0,
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<i16> {
        if input.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity((input.len() as f64 / self.ratio) as usize + 2);
        while self.pos < input.len() as f64 {
            let i = self.pos.floor() as isize;
            let frac = self.pos - i as f64;
            let s0 = if i < 0 { self.last } else { input[i as usize] };
            let s1 = if (i + 1) < input.len() as isize {
                input[(i + 1) as usize]
            } else {
                input[input.len() - 1]
            };
            let v = s0 as f64 + (s1 as f64 - s0 as f64) * frac;
            out.push((v.clamp(-1.0, 1.0) * 32767.0) as i16);
            self.pos += self.ratio;
        }
        self.pos -= input.len() as f64;
        self.last = input[input.len() - 1];
        out
    }
}
