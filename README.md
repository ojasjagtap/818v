# Recognizing Road Users by Motion (CMSC 818V)

Ojas Jagtap, Virali Kataria.

A self-driving car must know what each nearby object is. Cameras are good at this in daylight but
fail in the dark or when images blur. An object's motion is available in any lighting. We ask:

- **RQ1 (research question 1): physics vs. black box.** If we describe a track with physics
  (how tightly it can turn, whether it slides sideways, how hard it brakes), does that recognize
  road users better than a neural network that learns directly from the raw path? Especially in
  a city, or for a type of object, it never saw in training?
- **RQ2: motion vs. camera.** As camera images get darker or blurrier, at what point does motion
  become more reliable than the image, and does combining the two help?

Five classes throughout: vehicle, pedestrian, cyclist, motorcyclist, bus.

## Terms
- **Track / window**: an object's positions over time. We cut each moving object's track into one
  5-second window (50 positions, sampled 10 times per second, covering at least 3 m).
- **Macro-F1** (0 to 1, higher is better): the F1 score (balance of precision and recall)
  averaged over the five classes, so rare classes count as much as cars. We use it as the main
  score because 85% of objects are vehicles, which makes plain accuracy look good even when
  rare classes are missed.
- **Recall**: the share of a class's objects that the model labels correctly.
- **GBT (gradient-boosted trees)**: a standard classifier for a table of features.
- **GRU (gated recurrent unit)**: a neural network that reads the path point by point
  (our "black box").
- **Fusion**: combining two models' predicted probabilities.
- **AUROC** (0.5 = coin flip, 1 = perfect): how well a score separates two groups. Here it
  measures whether a model can tell "something it never saw" apart from known classes.
- **95% CI (confidence interval)**: the range the score would likely fall in on a different
  sample of the data.

## Data
- **MF**: Argoverse 2 *Motion Forecasting* dataset. About 225,000 11-second driving clips from
  6 US cities. Contains object tracks only (no images). Labels come from tracking software, not
  people. Used for RQ1.
- **Sensor**: Argoverse 2 *Sensor* dataset. 850 drives with human-labeled objects and 7 cameras
  around the car. Used for RQ2 (tracks plus image crops of each object) and to check our
  physics estimates against true object sizes.

## Layout
```
scripts/  setup_env.sh        Python environment and download tool (cluster login node)
          download_mf.sh      MF train + validation tracks (~34 GB)
          download_sensor.sh  Sensor labels, car poses, camera calibration, 1 image/second
          fetch_m1_sample.py  small local sample for the sanity checks
src/      common.py           shared constants, class names, window placement
          kinematics.py       smoothing and all physics / speed features
          m1_sanity.py        sanity checks (can motion reveal size? walking rhythm?)
          extract.py          MF clips -> 5-second windows
          aggregate.py        merge windows, compute features
          experiment.py       RQ1 train/test splits, decision rule, scores, novelty score
          train_gbt.py        GBT models: speed stats, physics, physics without noisy features
          train_seq.py        GRU, GRU + physics, GRU on smoothed paths
          fuse.py             combine the GRU's and physics model's predictions
          analyze.py          RQ1 tables and figures
          sensor_tracks.py    Sensor labels -> tracks and windows
          sensor_crops.py     cut a 128x128 image of each object from the cameras
          rq2.py              RQ2: image model, darkening/blurring, fusion, figures
```

## Results
Figures and supporting tables are in `results/` (`m1/` sanity checks, `rq1/`, `rq2/`).

### Data size (MF windows, `results/m2_counts.csv`)
| split | vehicle | pedestrian | bus | cyclist | motorcyclist |
|---|---|---|---|---|---|
| training | 1,605,183 | 197,974 | 44,408 | 29,037 | 7,962 |
| test (official validation split) | 200,580 | 24,271 | 5,547 | 3,547 | 1,016 |

Very unbalanced: 85% vehicles, 0.4% motorcyclists (79% of whom are in Miami).

### Sanity checks: what can motion tell us? (`results/m1/`)
- **Motion cannot reveal size.** A path's curvature depends on both the wheelbase (distance
  between axles) and the steering angle, and steering is not observed, so any wheelbase fits any
  path. Our size estimate exists for only 10–12% of windows, cannot tell buses from cars
  (AUROC 0.50), and barely follows true vehicle length on human-labeled data (correlation 0.15).
  See `wheelbase.png`. So our "physics" features describe how agile an object is, not how big.
- **The dataset's own velocity is unusable**: it lags the positions by 0.3–0.6 s, so we compute
  motion from positions.
- **Tracking noise gives away the class**: in MF, bus tracks are 2–3× as jittery as car tracks;
  in human-labeled data they are not. Some apparent "physics" is really a tracker quirk.
- **No walking or pedaling rhythm survives** in the 10-per-second tracks (`spectra.png`).

### RQ1: scores on the test split
"Buses called vehicle" = share of real buses the model labels as vehicle (lower is better).
"Cyclist ↔ motorcyclist" = share of cyclists labeled motorcyclist / share of motorcyclists
labeled cyclist.

| model | macro-F1 [95% CI] | accuracy | buses called vehicle | cyclist ↔ motorcyclist |
|---|---|---|---|---|
| GBT on speed statistics (baseline) | 0.453 [0.450, 0.456] | 0.898 | 0.78 | 0.00 / 0.04 |
| GBT on physics features | 0.539 [0.533, 0.545] | 0.915 | 0.59 | 0.01 / 0.08 |
| GBT on physics, noise-sensitive features removed | 0.517 [0.511, 0.523] | 0.915 | 0.66 | 0.01 / 0.09 |
| GRU on raw path (black box) | **0.635** [0.628, 0.641] | **0.939** | 0.69 | 0.03 / 0.14 |
| GRU + physics features | 0.541 [0.535, 0.546] | 0.931 | 0.71 | 0.01 / 0.09 |
| GRU on smoothed path | 0.613 [0.607, 0.619] | 0.930 | 0.64 | 0.04 / 0.10 |
| Fusion: average of GRU and physics GBT | 0.631 [0.624, 0.639] | 0.938 | 0.66 | 0.02 / 0.12 |

Differences between models (macro-F1, same test windows; an interval that includes 0 means no
real difference):

| comparison | difference [95% CI] |
|---|---|
| physics vs. speed statistics | +0.086 [0.081, 0.092] |
| physics vs. GRU | −0.095 [−0.103, −0.089] |
| GRU + physics vs. GRU | −0.094 [−0.100, −0.088] |
| GRU vs. GRU on smoothed path | +0.021 [0.016, 0.027] |
| physics vs. physics without noise-sensitive features | +0.022 [0.018, 0.027] |
| fusion vs. GRU | −0.003 [−0.008, 0.002] |

Takeaways: physics clearly beats plain speed statistics, but the GRU beats physics, and adding
physics to the GRU does not help. Both models lean slightly on tracking noise (smoothing it away
costs each about 0.02). The GRU's lead comes from rare classes: motorcyclist F1 0.44 vs 0.12 for
physics, cyclist 0.55 vs 0.44. Buses are hard for every model (F1 ≤ 0.30). The physics model relies
most on the tightest turn an object makes (shuffling it costs 0.19 macro-F1) and on its typical
acceleration (0.09); walking-rhythm features contribute almost nothing.

### RQ1: a city never seen in training
Each model is retrained without one city and tested on it. Numbers are how much macro-F1 drops
compared with cities it did see (higher = transfers worse).

| model | austin | dearborn | miami | palo-alto | pittsburgh | washington-dc | average drop | average score on unseen city |
|---|---|---|---|---|---|---|---|---|
| GBT on speed statistics | −0.002 | 0.031 | 0.001 | −0.030 | 0.008 | 0.020 | 0.005 | 0.447 |
| GBT on physics | 0.021 | 0.064 | −0.008 | 0.001 | 0.026 | 0.046 | 0.025 | 0.508 |
| Physics, noise features removed | 0.023 | 0.068 | −0.012 | 0.000 | 0.025 | 0.047 | 0.025 | 0.491 |
| GRU | 0.055 | 0.112 | 0.000 | 0.052 | 0.064 | 0.080 | 0.061 | 0.550 |
| GRU + physics | 0.023 | 0.082 | −0.007 | 0.004 | 0.031 | 0.047 | 0.030 | 0.516 |
| GRU on smoothed path | 0.054 | 0.100 | −0.001 | 0.055 | 0.060 | 0.078 | 0.058 | 0.538 |
| Fusion | 0.050 | 0.099 | −0.007 | 0.039 | 0.055 | 0.060 | 0.049 | **0.563** |

Takeaway: physics loses less than half as much as the GRU in a new city, though the GRU still
scores higher. Combining them (fusion) gives the best score on unseen cities.

### RQ1: a type of object never seen in training
Each model is retrained without one class. We then check whether it can flag those objects as
"unfamiliar" (AUROC; 0.5 = coin flip). In parentheses: the same, but only against the class the
unseen objects get confused with.

| model | bus unseen | cyclist unseen | motorcyclist unseen | pedestrian unseen |
|---|---|---|---|---|
| GBT on speed statistics | 0.68 (0.64) | 0.33 (0.27) | 0.56 (0.52) | 0.06 (0.22) |
| GBT on physics | **0.74** (0.72) | 0.37 (0.66) | 0.44 (0.41) | 0.52 (0.55) |
| Physics, noise features removed | 0.63 (0.61) | 0.45 (0.61) | 0.44 (0.42) | **0.66** (0.59) |
| GRU | 0.66 (0.64) | 0.51 (0.82) | **0.73** (0.71) | 0.36 (0.37) |
| GRU + physics | 0.72 (0.71) | **0.64** (0.80) | 0.56 (0.55) | 0.39 (0.34) |
| GRU on smoothed path | 0.65 (0.62) | 0.44 (0.80) | 0.69 (0.67) | 0.30 (0.35) |
| Fusion | 0.74 (0.72) | 0.43 (0.77) | 0.61 (0.58) | 0.42 (0.45) |

Takeaway: no model reliably notices new kinds of objects; scores below 0.5 mean the unseen class
looks *more* familiar than the known ones. Unseen objects get absorbed into the closest known
class: unseen buses are called vehicles 99–100% of the time, unseen pedestrians cyclists 84–90%.

### RQ2: motion vs. camera on clean images
849 drives, 16,297 tracked objects with 56,879 image crops (vehicle 9,139, pedestrian 6,365,
bus 475, cyclist 232, motorcyclist 86). Tested with 5-fold cross-validation: the data is split
into 5 parts by drive, and each part is tested by a model trained on the other 4.

- **Motion model**: the physics GBT from RQ1.
- **Camera model**: ResNet-18, a standard image classifier, trained on clean crops only.
- **Fixed-weight fusion**: combines both with one fixed amount of trust in the camera.
- **Quality-aware fusion**: sets how much to trust the camera for each object from how bright
  and sharp its image is.

| model | macro-F1 [95% CI] | accuracy [95% CI] |
|---|---|---|
| Motion | 0.470 [0.456, 0.484] | 0.893 [0.887, 0.899] |
| Camera | 0.758 [0.730, 0.785] | 0.942 [0.937, 0.946] |
| Fixed-weight fusion | 0.793 [0.766, 0.819] | 0.965 [0.961, 0.967] |
| Quality-aware fusion | **0.818** [0.791, 0.840] | **0.965** [0.961, 0.968] |

Motion alone recognizes vehicles (recall 0.90) and pedestrians (0.98) but almost never
motorcyclists (0.01) or buses (0.15); the camera handles those.

### RQ2: darker and blurrier images
Test images are made darker (×1/2 = half the light, down to 1/32, with realistic camera noise)
or blurrier (σ = blur radius in pixels on a 128-pixel crop). Motion is unaffected. The fusion
columns show the gain over whichever single model is better at that level. "Camera trust" is
how much weight quality-aware fusion gives the camera (0 to 1).

| image condition | motion | camera | fixed-weight fusion gain | quality-aware fusion gain | camera trust |
|---|---|---|---|---|---|
| clean | 0.470 | 0.758 | +0.035 | +0.060 | 0.71 |
| 2× darker | 0.470 | 0.634 | +0.038 | +0.063 | 0.60 |
| 4× darker | 0.470 | 0.506 | +0.090 | +0.102 | 0.38 |
| 8× darker | 0.470 | 0.402 | +0.035 | +0.058 | 0.23 |
| 16× darker | 0.470 | 0.324 | −0.006 | +0.018 | 0.16 |
| 32× darker | 0.470 | 0.256 | −0.022 | −0.004 | 0.13 |
| blur σ = 1 | 0.470 | 0.737 | +0.052 | +0.060 | 0.61 |
| blur σ = 2 | 0.470 | 0.659 | +0.081 | +0.079 | 0.52 |
| blur σ = 4 | 0.470 | 0.418 | +0.121 | +0.079 | 0.39 |
| blur σ = 8 | 0.470 | 0.132 | −0.021 | −0.022 | 0.26 |
| blur σ = 16 | 0.470 | 0.117 | −0.030 | −0.015 | 0.16 |

Takeaways (macro-F1): the camera falls below motion once images are 8× darker or blurred with
σ = 4. Measured by accuracy, it falls below motion even sooner in the dark (2× darker). Combining
helps most under mild to moderate damage, and quality-aware fusion stays close to the best option
everywhere because it learns to trust the camera less as images degrade. In the dark it keeps
accuracy near 0.92 at every level, while fixed-weight fusion drops to 0.84, below motion alone
(0.89). Full curves: `rq2/curves.csv` and `rq2/rq2_curves.png`.
