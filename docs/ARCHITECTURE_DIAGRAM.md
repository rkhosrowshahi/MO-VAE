# VQ-VAE2 + PixelCNN Prior Architecture

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VQ-VAE2 + PixelCNN SYSTEM                       │
└─────────────────────────────────────────────────────────────────────────┘

        STAGE 1: VQ-VAE2 Training              STAGE 2: Prior Training
   ┌────────────────────────────────┐      ┌─────────────────────────────┐
   │                                │      │                             │
   │  ┌──────┐    ┌─────────┐      │      │   ┌──────┐   ┌──────────┐  │
   │  │Image │───▶│ Encoder │      │      │   │Image │──▶│  Frozen  │  │
   │  │32x32 │    │         │      │      │   │      │   │ VQ-VAE2  │  │
   │  └──────┘    └────┬────┘      │      │   └──────┘   └─────┬────┘  │
   │                   │            │      │                    │        │
   │              ┌────▼────┐       │      │              ┌─────▼─────┐ │
   │              │ z_enc   │       │      │              │  z_top    │ │
   │              │ (cont.) │       │      │              │  z_bottom │ │
   │              └────┬────┘       │      │              │ (discrete)│ │
   │                   │            │      │              └─────┬─────┘ │
   │         ┌─────────▼──────┐    │      │                    │        │
   │         │  Quantization  │    │      │         ┌──────────▼──────┐│
   │         │  (VQ layers)   │    │      │         │   PixelCNN      ││
   │         └─────────┬──────┘    │      │         │   Prior         ││
   │                   │            │      │         └──────┬──────────┘│
   │              ┌────▼────┐       │      │                │           │
   │              │ z_quant │       │      │          ┌─────▼─────┐    │
   │              │(discrete)│      │      │          │Cross-Ent. │    │
   │              └────┬────┘       │      │          │   Loss    │    │
   │                   │            │      │          └───────────┘    │
   │              ┌────▼────┐       │      │                           │
   │              │ Decoder │       │      └───────────────────────────┘
   │              │         │       │
   │              └────┬────┘       │               STAGE 3: Generation
   │                   │            │      ┌─────────────────────────────┐
   │              ┌────▼────┐       │      │                             │
   │              │ Recons  │       │      │  ┌──────────────┐           │
   │              │  Image  │       │      │  │  PixelCNN    │           │
   │              └─────────┘       │      │  │  (sample)    │           │
   │                                │      │  └──────┬───────┘           │
   └────────────────────────────────┘      │         │                   │
                                            │    ┌────▼─────┐            │
                                            │    │ z_top    │            │
                                            │    │ z_bottom │            │
                                            │    │(discrete)│            │
                                            │    └────┬─────┘            │
                                            │         │                   │
                                            │    ┌────▼─────┐            │
                                            │    │  Frozen  │            │
                                            │    │ VQ-VAE2  │            │
                                            │    │ Decoder  │            │
                                            │    └────┬─────┘            │
                                            │         │                   │
                                            │    ┌────▼─────┐            │
                                            │    │Generated │            │
                                            │    │  Image   │            │
                                            │    └──────────┘            │
                                            └─────────────────────────────┘
```

## Hierarchical VQ-VAE2 Details

```
Input Image (3 x 32 x 32)
        │
        ▼
┌───────────────────┐
│  Bottom Encoder   │  Conv layers + ResBlocks
│  (stride 2x2)     │  Output: H/4 x W/4 = 8x8
└─────────┬─────────┘
          │
          ├─────────────────────────┐
          │                         │
          ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  Top Encoder    │       │  Project to     │
│  (stride 2x2)   │       │  Embedding Dim  │
│  Output: 4x4    │       └────────┬────────┘
└────────┬────────┘                │
         │                          │
         ▼                          ▼
┌─────────────────┐       ┌─────────────────┐
│  VQ-Top         │       │  VQ-Bottom      │
│  Codebook: 512  │       │  Codebook: 512  │
│  z_top: 4x4     │       │  z_bottom: 8x8  │
└────────┬────────┘       └────────┬────────┘
         │                          │
         │                          │
         ▼                          │
┌─────────────────┐                │
│  Top Decoder    │                │
│  (upsample 2x)  │                │
│  Output: 8x8    │                │
└────────┬────────┘                │
         │                          │
         └──────────┬───────────────┘
                    │ Concatenate
                    ▼
          ┌──────────────────┐
          │  Bottom Decoder  │
          │  (upsample 2x2)  │
          │  Output: 32x32   │
          └─────────┬────────┘
                    │
                    ▼
          Reconstructed Image
```

## PixelCNN Prior Architecture

### Single PixelCNN Block

```
Input: [B, H, W] discrete codes
        │
        ▼
┌─────────────────────┐
│  Embedding Layer    │  codes → continuous [B, H, W, D]
└─────────┬───────────┘
          │ Permute to [B, D, H, W]
          ▼
┌─────────────────────┐
│  MaskedConv2d (A)   │  Type A: excludes center pixel
│  kernel_size=7      │  7x7 receptive field
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Gated ResBlock 1   │
├─────────────────────┤
│  • Conv 1x1         │  [channels → channels/2]
│  • MaskedConv 3x3(B)│  Type B: includes center
│  • Gated Activation │  tanh(W_f*x) ⊙ sigmoid(W_g*x)
│  • Conv 1x1         │  [channels/2 → channels]
│  • Residual Add     │
└─────────┬───────────┘
          │
          ▼
     [15 more blocks...]
          │
          ▼
┌─────────────────────┐
│  Output Layers      │
├─────────────────────┤
│  • ReLU             │
│  • Conv 1x1         │  [channels → channels]
│  • ReLU             │
│  • Conv 1x1         │  [channels → num_embeddings]
└─────────┬───────────┘
          │
          ▼
  Logits [B, K, H, W]
```

### Hierarchical PixelCNN

```
┌──────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL PIXELCNN                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────┐                                 │
│  │    Prior Top           │  Models P(z_top)                │
│  │    PixelCNN            │                                 │
│  │                        │  Input:  z_top [B, 4, 4]        │
│  │    (unconditional)     │  Output: logits [B, 512, 4, 4] │
│  └───────────┬────────────┘                                 │
│              │                                               │
│              │ (for generation)                              │
│              ▼                                               │
│      ┌──────────────┐                                        │
│      │  Embedding   │                                        │
│      │  Upsample 2x │  z_top [4,4] → [8,8]                  │
│      └──────┬───────┘                                        │
│             │                                                │
│             │ Conditioning                                   │
│             ▼                                                │
│  ┌────────────────────────┐                                 │
│  │    Prior Bottom        │  Models P(z_bottom | z_top)     │
│  │    PixelCNN            │                                 │
│  │                        │  Input:  z_bottom [B, 8, 8]     │
│  │    (conditional)       │          + z_top_up [B, D, 8,8] │
│  │                        │  Output: logits [B, 512, 8, 8]  │
│  └────────────────────────┘                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Masked Convolution Mechanism

### Type A (First Layer)

```
Receptive Field (3x3 example):
┌───┬───┬───┐
│ ✓ │ ✓ │ ✓ │  ✓ = can see
├───┼───┼───┤
│ ✓ │ ✗ │   │  ✗ = current pixel (masked out)
├───┼───┼───┤    (blank) = future pixels
│   │   │   │
└───┴───┴───┘

Mask Pattern:
[1  1  1]
[1  0  0]  Current pixel excluded
[0  0  0]
```

### Type B (Subsequent Layers)

```
Receptive Field (3x3 example):
┌───┬───┬───┐
│ ✓ │ ✓ │ ✓ │  ✓ = can see
├───┼───┼───┤
│ ✓ │ ✓ │   │  Current pixel included
├───┼───┼───┤    (blank) = future pixels
│   │   │   │
└───┴───┴───┘

Mask Pattern:
[1  1  1]
[1  1  0]  Current pixel included
[0  0  0]
```

## Gated Activation Mechanism

```
Input: x [B, C, H, W]
        │
        ├────────────┬────────────┐
        │            │            │
        ▼            ▼            │
   ┌────────┐  ┌────────┐        │
   │ Conv_f │  │ Conv_g │        │
   │(feature)│  │ (gate) │        │
   └────┬───┘  └────┬───┘        │
        │            │            │
        ▼            ▼            │
   ┌────────┐  ┌────────┐        │
   │  tanh  │  │sigmoid │        │
   └────┬───┘  └────┬───┘        │
        │            │            │
        └─────┬──────┘            │
              │                   │
              ▼                   │
          ┌───────┐               │
          │  ⊙    │  Element-wise │
          │(mul)  │  multiply     │
          └───┬───┘               │
              │                   │
              └───────┬───────────┘
                      │
                      ▼
                 ┌────────┐
                 │   +    │  Residual
                 │  (add) │  connection
                 └────────┘
                      │
                      ▼
                   Output
```

## Autoregressive Generation Order

```
Raster Scan Order (4x4 example):

┌────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │
├────┼────┼────┼────┤
│ 5  │ 6  │ 7  │ 8  │
├────┼────┼────┼────┤
│ 9  │10  │11  │12  │
├────┼────┼────┼────┤
│13  │14  │15  │16  │
└────┴────┴────┴────┘

For each position i:
  1. Compute P(z_i | z_{1..i-1})
  2. Sample z_i ~ Categorical(P)
  3. Move to next position

Total steps: H × W (16 for 4×4, 64 for 8×8)
```

## Data Dimensions Throughout

```
Input Image: [B, 3, 32, 32]
        ↓ Encoder Bottom
Bottom Features: [B, 256, 8, 8]
        ↓ Split
        ├─→ Encoder Top → [B, 256, 4, 4] → Project → [B, 64, 4, 4]
        │         ↓ VQ-Top
        │   z_top indices: [B, 4, 4]  (discrete: 0-511)
        │   z_top quant: [B, 64, 4, 4] (continuous embeddings)
        │
        └─→ Project → [B, 64, 8, 8]
                ↓ VQ-Bottom
          z_bottom indices: [B, 8, 8]  (discrete: 0-511)
          z_bottom quant: [B, 64, 8, 8] (continuous embeddings)

For PixelCNN Training:
  Input: z_top [B, 4, 4] or z_bottom [B, 8, 8] (integer codes)
  Output: logits [B, 512, 4, 4] or [B, 512, 8, 8]
  Loss: CrossEntropy(logits, true_codes)

For Generation:
  Sample: z_top [B, 4, 4], z_bottom [B, 8, 8] (from PixelCNN)
  Embed: → [B, 64, 4, 4] and [B, 64, 8, 8]
  Decode: → [B, 3, 32, 32]
```

## Comparison: With vs Without Prior

### Without PixelCNN Prior (Random Sampling)

```
┌──────────────────┐
│  Random Uniform  │
│   Sampling       │
│   P(z) = 1/K     │  K = codebook size
└────────┬─────────┘
         │
         ▼
    ┌────────┐
    │ z_top  │  Random codes
    │z_bottom│  (no structure)
    └────┬───┘
         │
         ▼
  ┌─────────────┐
  │  VQ-VAE2    │
  │  Decoder    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Noisy      │  ✗ Low quality
  │  Image      │  ✗ No coherence
  └─────────────┘  ✗ Not realistic
```

### With PixelCNN Prior (Learned Distribution)

```
┌──────────────────┐
│  PixelCNN Prior  │
│  P(z) learned    │
│  from data       │
└────────┬─────────┘
         │
         ▼
    ┌────────┐
    │ z_top  │  Realistic codes
    │z_bottom│  (learned structure)
    └────┬───┘
         │
         ▼
  ┌─────────────┐
  │  VQ-VAE2    │
  │  Decoder    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Sharp      │  ✓ High quality
  │  Image      │  ✓ Coherent
  └─────────────┘  ✓ Realistic
```

## Training Flow

```
STAGE 1: VQ-VAE2 (e.g., 100 epochs)
─────────────────────────────────
Epoch 1:  [####----------------] Loss: 0.8523
Epoch 20: [########------------] Loss: 0.4521
Epoch 50: [############--------] Loss: 0.2145
Epoch 100:[####################] Loss: 0.1234
✓ Save checkpoint: vqvae2_final.pth

STAGE 2: PixelCNN Prior (e.g., 100 epochs)
──────────────────────────────────────────
Load VQ-VAE2 checkpoint ✓
Freeze VQ-VAE2 weights ✓

Epoch 1:  [####----------------] Loss: 6.2345 (NLL)
Epoch 20: [########------------] Loss: 4.8521
Epoch 50: [############--------] Loss: 3.6145
Epoch 100:[####################] Loss: 2.8234
✓ Save checkpoint: prior_final.pth

STAGE 3: Generation
───────────────────
Load both checkpoints ✓
Generate 64 samples:
  Progress: [████████████████████] 64/64
  Time: 45.3 seconds
✓ Saved to: samples.png
```

## Memory Requirements

```
                    GPU Memory (GB)
Component           Training    Inference
────────────────────────────────────────
VQ-VAE2 alone         ~4 GB      ~1 GB
PixelCNN Prior        ~6 GB      ~2 GB
Both (generation)       -        ~3 GB
────────────────────────────────────────
Peak (prior train)   ~8 GB        -
Peak (generation)      -        ~3 GB

Batch Size Impact (Prior Training):
  BS=64:   ~4 GB
  BS=128:  ~6 GB
  BS=256:  ~10 GB
```

## Performance Metrics

```
Model Component       Params      Latency       Quality
──────────────────────────────────────────────────────
VQ-VAE2 Encoder      ~3M         5ms/sample      -
VQ-VAE2 Decoder      ~3M         5ms/sample      -
PixelCNN Top         ~8M         2s/sample     High
PixelCNN Bottom      ~8M         8s/sample     High
──────────────────────────────────────────────────────
Total System         ~22M        ~10s/sample   High

Generation Time Breakdown (per sample):
├─ PixelCNN Top:    2s  (16 sequential steps for 4×4)
├─ PixelCNN Bottom: 8s  (64 sequential steps for 8×8)
└─ VQ-VAE2 Decode:  5ms (parallel operation)
Total:              ~10s
```

## Key Takeaways

1. **Two-Stage Training**: VQ-VAE2 first, then PixelCNN prior
2. **Hierarchical Structure**: Top (4×4) → Bottom (8×8) conditioning
3. **Autoregressive**: Sequential generation (inherently slow)
4. **Quality Boost**: Prior enables realistic generation
5. **Trade-offs**: Slow inference but high quality

## Next Steps

- See `QUICKSTART_PIXELCNN.md` for getting started
- See `PIXELCNN_PRIOR_README.md` for detailed documentation
- See `examples/pixelcnn_prior_example.py` for code examples

