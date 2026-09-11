# FLOCK//MIND - Neuroevolution Flight Lab

Watch a population of 200 neural networks learn to fly, live in your browser.
No training data, no backprop - just selection, crossover, and mutation
playing Flappy Bird at up to 64x speed.

**Live:** https://jacobegarcia.github.io/flock-mind/

## What you're looking at

- **FLIGHT CHAMBER** - the whole population flying at once. The amber-ringed
  bird is the current leader; ghosts are the rest of the flock.
- **TELEMETRY** - generation counter, alive count, gate score, best/mean
  fitness, sim clock, and a live checksum of the champion genome.
- **FITNESS BY GENERATION** - best and mean fitness per generation. Click the
  chart to inspect any generation.
- **LEADER BRAIN** - the live 5-8-1 network of the leading bird. Inputs
  (altitude, velocity, pipe distance, gap top/bottom) light up cyan, hidden
  units glow amber, and the FLAP output fires in real time. Amber edges are
  excitatory, cyan edges inhibitory.
- **GENERATION ARCHIVE** - every generation's champion (genome + world seed)
  is archived. Scrub the timeline and hit REPLAY CHAMPION to watch a past
  champion fly its exact original course, deterministically.

## Controls

| Input | Action |
| --- | --- |
| `SPACE` | pause / resume |
| `1` - `8` | speed: 1x, 2x, 4x, 8x, 16x, 32x, 64x, MAX |
| `R` | reset evolution (new random seed) |
| `L` | return to live from an archive replay |

## How it works

- Genome: 57 floats - a fixed 5-8-1 feedforward network (tanh hidden).
- Fitness: distance flown + 1000 per gate cleared.
- Selection: 2 elites cloned, the rest tournament-selected (k=3) parents with
  uniform crossover and gaussian mutation (rate 0.12, sigma 0.35).
- Each generation's pipe field is seeded and stored, so champion replays are
  bit-exact.

Vanilla JS + Canvas. Zero dependencies, zero build step. Static site.
