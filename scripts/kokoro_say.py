#!/usr/bin/env python3
"""Reference Kokoro TTS sidecar for voice-assistant's `tts=cmd` engine.

Contract (what voice-assistant's Engine::Cmd expects):
  - read the utterance text from STDIN (until EOF),
  - synthesize and PLAY it (this process owns playback),
  - exit when done. A single process, so voice-assistant can kill it to
    interrupt speech ("停" / barge-in).

This is a RECIPE to adapt, not a guaranteed-working drop-in: install the deps
and point the model paths at your download. Chinese needs the zh model + a
Chinese G2P (misaki[zh]).

Setup (one time):
  pip install kokoro-onnx misaki[zh] sounddevice numpy
  # models (Chinese): https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
  #   or the general v1.0 onnx + voices bin from onnx-community / thewh1teagle
  # put the .onnx and voices .bin somewhere and set the env vars below.

Wire it into voice-assistant:
  voice-assistant setup   # 5) 语音回复 -> 3) 自定义命令
  # command:  python3 /abs/path/scripts/kokoro_say.py
  # (or set VA_TTS=cmd and VA_TTS_CMD="python3 /abs/path/scripts/kokoro_say.py")

Env (override defaults):
  KOKORO_MODEL   path to kokoro onnx        (default: ~/.voice-assistant/kokoro/kokoro.onnx)
  KOKORO_VOICES  path to voices .bin        (default: ~/.voice-assistant/kokoro/voices.bin)
  KOKORO_VOICE   voice id                   (default: zf_xiaoxiao — a zh voice)
  KOKORO_LANG    language code              (default: zh)
  KOKORO_SPEED   speed multiplier           (default: 1.0)
"""
import os
import sys


def main() -> int:
    text = sys.stdin.read().strip()
    if not text:
        return 0

    home = os.path.expanduser("~")
    model = os.environ.get("KOKORO_MODEL", f"{home}/.voice-assistant/kokoro/kokoro.onnx")
    voices = os.environ.get("KOKORO_VOICES", f"{home}/.voice-assistant/kokoro/voices.bin")
    voice = os.environ.get("KOKORO_VOICE", "zf_xiaoxiao")
    lang = os.environ.get("KOKORO_LANG", "zh")
    speed = float(os.environ.get("KOKORO_SPEED", "1.0"))

    try:
        import sounddevice as sd
        from kokoro_onnx import Kokoro
    except ImportError as e:
        sys.stderr.write(
            f"[kokoro_say] missing dep: {e}\n"
            "  pip install kokoro-onnx misaki[zh] sounddevice numpy\n"
        )
        return 1

    if not (os.path.exists(model) and os.path.exists(voices)):
        sys.stderr.write(
            f"[kokoro_say] model files not found:\n  {model}\n  {voices}\n"
            "  download them and/or set KOKORO_MODEL / KOKORO_VOICES.\n"
        )
        return 1

    kokoro = Kokoro(model, voices)
    # kokoro-onnx returns (samples: float32 ndarray, sample_rate: int).
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    sd.play(samples, sr)
    sd.wait()  # block until playback finishes; killing this process stops it
    return 0


if __name__ == "__main__":
    sys.exit(main())
