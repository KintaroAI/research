# Task 00077: A World of Sound

**Date:** 2026-11-01
**Status:** Planned, not started (the owner: keep the idea written; real time does not matter for now)

## Context

- The universal model has lived on pixels, text and a camera with no change for the world. Sound is the next test of that: if it can be shown as a grey frame, the model should need nothing new.
- Speech models commonly start from a log-mel spectrogram: time on one axis, frequency on the other, brightness for loudness. That is already an image, so it fits the existing token format.
- The goal stays universal > prediction > instincts: no special path for sound. If sound shows a limit of the model, the model changes in a way that works the same for every world.

## Design

**The world.** `world/sound_world.py`, `env.kind = sound`, with CircleWorld's interface as `text_world.py` has it: moves ignored, drives 0, no episodes. A frame is a 64 by 64 grey image: 64 mel bands by the last 64 columns, newest on the right. Each tick scrolls it by `env.hop` columns.

- One column is 20 ms (window 40 ms), so 60k ticks at `hop = 1` hear 20 minutes of audio. Real time is not a goal here.
- `hop = 1` makes most of each frame a copy of the last. `hop = 8` brings in a whole column of patches per tick (160 ms). Try both first: it is the only open choice in the world.
- Grey levels: log-mel power, clipped to a fixed range and mapped to 0–255, the same range for every clip.

**Preparation.** A script that uses numpy and the standard `wave` module, no new dependency, writes each dataset once to a uint8 array in `~/data/research/visual-motor-agent/sound/`. Clips are joined with a short silence into one stream, so that a life is one long hearing, as with the text corpus. Audio stays out of the repo. Downloads are untrusted data: each in its own empty directory, read with `python -I`.

**Datasets** (sizes from memory: check on download).

| dataset | use | contents | size | licence |
|---|---|---|---|---|
| Free Spoken Digit Dataset | pipeline check | 3,000 short recordings of digits, a few speakers | under 20 MB | CC BY-SA |
| LJ Speech | main run | one speaker reading, about 24 hours in 13,100 clips, with transcripts | about 2.6 GB | public domain |
| Mini LibriSpeech | later: other voices | many speakers reading | about 330 MB | CC BY |

LJ Speech is the Tiny Shakespeare of this world: one voice, steady quality, one continuous stream. An hour of it covers a lifetime, so hold out whole chapters (the highest `LJ0NN` ids) as unseen audio.

**Config.** `configs/U_sound.json`: `U_default.json`'s model with the sound world. The hunger instinct has nothing to work on here, so it stays off, as in the text world.

**Listening to it.** A tool that turns a spectrogram back into a waveform (Griffin-Lim, numpy) and writes a WAV, so that the model's dream can be heard, not only scored: real and dreamed side by side.

## Measures

- `eval_predict.py`'s held-out nats per value, on unseen audio.
- **The baseline to beat:** repeat the last column, scrolled by `hop`. Speech changes slowly, so this will look good, and there is no classical reference like the character GPT. Plain "repeat the last frame" is not a fair baseline when the frame scrolls.
- The dream: how many steps of free prediction stay better than the baseline, and whether it still sounds like speech.

## The decision, set before the results

The sound world works if, on held-out audio at `hop` chosen by the first run, the model's prediction beats the baseline's interval over 3 seeds on Free Spoken Digit, and then on LJ Speech. Otherwise report which limit showed, and change the model for every world, not for sound.

## Verification

Tests as for the text world: `world/` interface and determinism, the scroll (a frame shifted by `hop` equals the next frame except for the new columns), splits that share no clip, the mel pipeline against a pure tone (a bright line in the right band), and Griffin-Lim's round trip on a known clip. The model's suite is untouched: no model code changes.

## Later: speaking (the owner, 2026-11-01)

In a language model the input and the output are one channel: the next token is both the target and the next input. In natural intelligence they are different: we learn to predict speech by ear, while the output is a separate set of muscles that nobody trains us to move to repeat the next word. Speech emerges almost in parallel with listening, and the only loop is hearing ourselves on top of the world. The experiment to think about: a model that learns to predict the sound it hears and, with a separate motor output and no loop built for it, learns to make sound itself.

## Results

Pending.
