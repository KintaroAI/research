# Temporal Signal Matrix: Modalities & Value Representation

## Context

The thalamus sorter discovers spatial structure from temporal correlations. Each neuron has an activity trace over time — the temporal matrix `(N, T)`. Experiments ts-00009 proved this works with synthetic Gaussian-smoothed noise. Moving toward biological realism means supporting multiple sensory modalities with realistic signal properties.

## What should the values represent?

### Option A: Raw sensor activations
- Vision neurons get pixel intensities (0-255)
- Audio neurons get frequency bin amplitudes
- Touch neurons get pressure values
- Different scales per modality, but Pearson correlation normalizes this away
- Cross-modal neurons have zero correlation unless temporal coincidence exists

### Option B: Normalized firing rates (0-1)
- Everything on the same scale
- Each value = "how active is this neuron right now"
- Biologically, this is what downstream areas observe — not raw sensor values, but neural firing rates from sensory areas
- Clean for Pearson correlation

### Option C: Binary spikes (0/1)
- Most biologically realistic at single-neuron level
- Pearson correlation on binary vectors is noisy — needs very long T for reliable estimates
- ts-00009 showed T=50 was already too noisy with continuous values; binary would be worse

### Decision: Normalized firing rates (Option B)

Reasons:
1. It's what the thalamus actually sees — neural firing rates, not raw pixels
2. Pearson correlation works well on continuous values (vs binary)
3. Scale-invariant across modalities — a vision neuron at 0.8 and an audio neuron at 0.8 are both "strongly active"
4. The spatial correlation structure comes from *how* the values are generated (receptive field overlap), not from the values themselves

## Multimodal signal generation

Each modality has its own spatial correlation structure:

- **Vision (retinotopic):** 2D spatial smoothing. Nearby pixels in visual field see similar stimuli. Gaussian blur with sigma proportional to receptive field size.
- **Audio (tonotopic):** 1D spatial smoothing along frequency axis. Adjacent frequency-tuned neurons correlate because natural sounds have spectral continuity.
- **Touch (somatotopic):** 2D spatial smoothing over body surface map. Adjacent skin areas get similar pressure/temperature.

### Cross-modal correlations

Within a modality, correlation comes from receptive field overlap (spatial smoothing). Across modalities, correlation comes from **temporal coincidence** — seeing and hearing the same event at the same time.

- A clap: visual motion neurons and audio onset neurons fire simultaneously
- Walking on gravel: visual texture neurons, audio crunch neurons, and foot pressure neurons co-activate

This creates a richer correlation structure: within-modality correlations are strong and continuous (spatial smoothing), cross-modal correlations are sparse and event-driven (temporal coincidence).

### Open questions

- How to model cross-modal temporal coincidence? Shared "event" signals that activate subsets of neurons across modalities?
- Should modalities occupy separate regions of the neuron array, or be interleaved?
- What's the right ratio of within-modal vs cross-modal correlation strength?
- Does the skip-gram learner naturally separate modalities into clusters, or does it need help?
