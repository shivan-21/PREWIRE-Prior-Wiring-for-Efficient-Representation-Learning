# How to read the figures

A plain-language guide to every figure in `figures/`: how it was made and what it shows. File names end in the date the figure was made (`_YYYYMMDD`). Each figure exists as a PDF (for print) and a PNG (for slides).

## The setup in one paragraph

We take one network (VGG-11, 8 layers) and let it learn from 50,000 CIFAR-10 photos **without any labels**, using one of four methods:

- **LPL**: the brain-like rule we study. It assumes that what you see now looks like what you saw a moment ago.
- **Oja**: a classic, simpler brain-like rule that only learns the most common patterns.
- **SimCLR**: a strong, standard AI method that is not brain-like.
- **Random init**: the network is never trained. This is the baseline.

Then we test how useful each network is when only a few labelled photos (10 to 50,000) are available. Every test is scored on 10,000 separate test photos. Most results are repeated with 3 seeds: 3 independent repeats, each with different randomness and a different set of labelled photos.

Two tests are used throughout:

- **Linear probe.** The network is **frozen**. Only a simple classifier on top of it is trained on the labelled photos. This measures how good the learned features are as they stand.
- **Fine-tuning.** The **whole network** is trained further on the labelled photos. This measures how good a starting point the pre-training provides.

---

## `temporal_clips_preview`: what the fake "videos" look like

**How it was made.** Each row turns one photo (left) into an 8-frame clip (t = 0 to 7). Each frame is an edit of the photo: a different crop or zoom, mirror flip, colour, brightness, greyscale or blur. These are the same edits Zenke's LPL code uses. The knob **ρ (rho)** sets how much one frame resembles the previous one:

- ρ = 0: every frame is an independent random edit.
- ρ = 0.95: the edits drift very slowly.

All rows start from the same first frame, so you can see them diverge.

**Takeaway.** CIFAR-10 has no real video, so this is how we create "time": a stand-in for an eye drifting over an object while the lighting changes. At ρ = 0 consecutive frames jump around. At ρ = 0.95 they barely change.

## `training_progress`: how the methods learn during pre-training

**How it was made.** Every 2,000 training steps we briefly froze the network and checked it: a simple classifier was trained on the last layer's features of 10,000 labelled photos and scored on the test photos. This check never changes the network. Lines are the average of 3 seeds, and the shaded band shows the seed-to-seed spread. The dashed line is the untrained network.

**Takeaway.**

- SimCLR learns fast and plateaus at about 79%.
- LPL climbs steadily to about 61%. It dips below the untrained network at first.
- Oja creeps up to about 50%.
- All three have levelled off by the end, so the training budget (35,000 steps) was enough.

## `data_efficiency_probe`: frozen features vs number of labels

**How it was made.** For each method and each label count (10, 20, 50 … 50,000; x-axis, log scale), a simple classifier was trained on the frozen last layer and scored on the test photos. Lines are the average of 3 seeds, and the band shows the spread. The dotted blue line is LPL trained with **Zenke's original code**, scored the same way.

**Takeaway.**

- **Our LPL matches Zenke's own code**, so our reimplementation is faithful.
- **LPL's features beat the untrained network's at every label count.** With just 1,000 labels LPL (49.8%) already beats the untrained network given all 50,000 labels (48.8%).
- LPL also beats Oja everywhere.
- SimCLR is far ahead: that is the price of being brain-like.

## `data_efficiency_finetune`: whole network trained vs number of labels

**How it was made.** As above, but now the whole network is trained further on the labelled photos: 30 passes over them (at least 1,000 training steps), using one fixed recipe for every method. The untrained network here simply trains from scratch.

**Takeaway.**

- **When labels are scarce, LPL helps a lot.**
  - With 100 labels, LPL reaches 44%. Training from scratch needs about 570 labels to match that (≈6× more).
  - With 1,000 labels, LPL reaches 59%. From scratch needs about 3,800 (≈4× more).
  - Up to about 1,000 labels, LPL also beats Oja.
- **LPL's head start shrinks as labels grow, and at 50,000 labels it turns into a handicap**: 78.7% vs 81.0% from scratch. This holds for every seed.
- **Oja helps at every label count and overtakes LPL from about 5,000 labels** (85.8% vs 78.7% at 50,000). This also holds for every seed.
- SimCLR is far ahead throughout: with 100 labels it matches what from scratch needs about 4,900 labels for.
- One possible reading: LPL's built-in assumption is a head start when data is scarce. But its features have learned to ignore things like colour and position, and that can get in the way when plenty of data is available. Oja's simpler features are a more neutral starting point.
- This is the knowledge–data tradeoff in action: **a prior is worth most when data is scarce, and can even cost you when data is plentiful.**
- Note: an earlier version of this figure had a noisy from-scratch curve because of a bug, fixed on 2026-11-19; see `docs/update.md`.

## `layer_probe`: which layer holds the useful information

**How it was made.** The same frozen-feature test as above, but run on each of the 8 layers separately (x-axis), at 100, 1,000 and 50,000 labels (one panel each). Lines are the average of 3 seeds.

**Takeaway.**

- For SimCLR, accuracy rises layer by layer: each layer builds on the one before it.
- For LPL, accuracy rises through the early layers and **levels off around layer 5**, then drifts slightly down. Each LPL layer learns on its own, so the later layers add less.
- Oja gains a little up to layer 3–5, then loses it again. The untrained network gets slightly worse with depth.
- Practical point: for LPL, a middle layer is about as useful as the last one.

## `ablation_probe`: what LPL actually needs

**How it was made.** Each bar is one version of LPL, frozen and tested on its last layer, at 100, 1,000 and 50,000 labels (one panel each). The versions:

- **Two views (standard LPL)**: two random edits of the same photo.
- **Clips**: consecutive frames of the fake videos above, at different smoothness ρ.
- **Clips, frame order shuffled**: the same smooth clips with their frames put in random order. Paired frames are then less similar on average.
- **Partner from another image / another clip**: each frame is paired with a frame of a *different* photo. This breaks the "same object" assumption.

Oja and the untrained network are shown for reference. Error bars show the spread for the 3-seed runs; the clip runs and the shuffled-pairing runs were done once each.

**Takeaway.**

- **Pairing different photos is a disaster.** LPL then falls *below the untrained network*. "Same object, slightly changed" is the essential ingredient.
- **Smooth clips beat independent frames**: time, as a source of similar pairs, helps.
- **Shuffling the frame order costs little**, and not consistently. LPL only ever compares two frames, so what matters is how similar the paired frames are, not the order.

## `temporal_rho`: how smooth should time be?

**How it was made.** The frozen last-layer test for the clip versions of LPL, plotted against smoothness ρ (x-axis), with one line per label count. The crosses show the ρ = 0.8 clips with their frame order shuffled.

**Takeaway.**

- **There is a sweet spot.** Accuracy rises from ρ = 0 to ρ = 0.8, then drops at ρ = 0.95.
- If consecutive frames are too different, they don't clearly show the same thing. If they're nearly identical, there is nothing left to learn from the change.
- The crosses sit a little below ρ = 0.8, close to the ρ = 0.5 points. That matches the idea that shuffling the order simply makes paired frames less similar.
- These are single runs, so small differences (about 1–2 points) may be noise.
