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

## Vision: RGB to firing rates

### Neurons per pixel

**Option 1: Grayscale (1 neuron per pixel)** ← starting here
- Firing rate = luminance / 255
- Simplest, matches ts-00009 setup (one neuron per grid cell)
- Loses color information but sufficient to validate the pipeline

**Option 2: RGB (3 neurons per pixel)**
- Each pixel → R, G, B neurons
- Firing rate = channel intensity / 255
- 80x80 image → 19,200 neurons (3x)
- Sorter should discover both spatial proximity (co-located channels correlate) and channel grouping

**Option 3: Biologically inspired channels**
- Luminance (L+M), red-green opponency (L-M), blue-yellow opponency (S-(L+M))
- Plus ON/OFF channels (responds to increase vs decrease)
- More realistic but over-engineered for current stage

### Temporal variation (where does the signal come from?)

Static image = one fixed firing rate per neuron. Correlation over time is undefined. Need temporal variation to create meaningful correlation structure.

**Option A: Random crops (simulated saccades)** ← plan
- Each timestep, take a random crop/position from a larger image
- Neurons see different content each timestep
- Nearby neurons see similar crops (overlapping receptive fields) → temporal correlation
- Simulates biological eye movements (saccades) that constantly shift the visual input
- Simple to implement: random (dx, dy) offset per timestep, wrap or clamp at edges

**Option B: Video stream**
- Natural temporal variation from real video
- Most realistic — objects move, lighting changes, camera moves
- Nearby pixels see similar temporal patterns (same object, same motion)
- Requires video data source

**Option C: Random images from dataset**
- Each timestep is a completely different image
- Correlation comes from statistical regularities across images (natural image statistics)
- Weaker signal than crops — two nearby pixels seeing unrelated images have weak correlation
- Would need very large T to build up reliable correlation estimates

**Option D: Gaussian noise modulated by image (ts-00009 approach)**
- Synthetic signal: Gaussian-smoothed noise × pixel intensity
- Already proven to work, but not image-aware beyond static weighting
- The correlation structure comes from the Gaussian blur, not from the image content

### Decision: Grayscale + random crops

Start with simplest viable approach:
- 1 neuron per pixel (grayscale)
- Temporal signal from random crops of a larger source image
- Firing rate at timestep t = pixel intensity at shifted position / 255
- Nearby neurons see similar shifts → correlated firing patterns
- Validates the pipeline before adding RGB or other modalities

### Open questions

- How to model cross-modal temporal coincidence? Shared "event" signals that activate subsets of neurons across modalities?
- Should modalities occupy separate regions of the neuron array, or be interleaved?
- What's the right ratio of within-modal vs cross-modal correlation strength?
- Does the skip-gram learner naturally separate modalities into clusters, or does it need help?
- Crop size vs grid size: should the source image be 2x the grid? 4x? How much shift per timestep?
- Wrap vs clamp at edges: wrapping creates toroidal topology, clamping creates edge effects
