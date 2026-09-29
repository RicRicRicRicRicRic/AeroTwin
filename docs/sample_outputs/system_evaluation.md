# AeroTwin AI — System Evaluation (Phase 7)

- **Generated:** 2026-09-29T12:17:19.728811+00:00
- **Outcome:** `completed`
- **Seed:** 42
- **Stage wall-time total:** 2.2152 s
- **Weights source:** `synthetic_random_init`
- **Data dir:** `C:\Users\ricmi\AppData\Local\Temp\aerotwin_eval_20260929T121713Z`

## Pipeline stages

| # | Stage | Status | Wall (s) | Backend (s) | Key metric |
|---|-------|--------|---------:|------------:|------------|
| 1 | `video_import` | completed | 0.050 | — | 24f @ 10fps 64x48 |
| 2 | `frame_extraction` | completed | 0.092 | 0.022 | 8 frames written |
| 3 | `weights_calibration` | completed | 1.167 | — | {"status": "calibrated", "frames_used": 3, "crack": {"head_scale": 4.0, "background_bias": -0... |
| 4 | `material_segmentation` | completed | 0.112 | 0.063 | mean 3.7 ms/frame (total 0.029 s) |
| 5 | `element_detection` | completed | 0.122 | 0.067 | mean 4.0 ms/frame |
| 6 | `crack_mapping` | completed | 0.159 | 0.108 | 8 frames w/ cracks |
| 7 | `cross_frame_aggregation` | completed | 0.117 | 0.032 | dedup 87.5% (8→1 crack obs) |
| 8 | `seismic_assessment` | completed | 0.107 | — | score 55.13/100 — Substantial |
| 9 | `report_generation` | completed | 0.287 | — | 23979 bytes written |

## Cross-frame dedup

| Metric | Value |
|--------|------:|
| raw_crack_observations | 8 |
| unique_crack_defects | 1 |
| crack_dedup_ratio | 0.875 |
| crack_dedup_reduction | 7 |
| mean_observations_per_crack | 8.0 |
| raw_element_observations | 16 |
| unique_element_instances | 2 |
| frames_analyzed | 8 |
| iou_threshold | 0.3 |
| max_frame_gap | 10 |

## Seismic assessment

- **Vulnerability score:** 55.13/100 — **Substantial**

| Factor | Value | Weight | Contribution |
|--------|------:|-------:|-------------:|
| age | 0.5 | 0.12 | 0.06 |
| construction | 0.15 | 0.14 | 0.021 |
| stories | 0.555556 | 0.06 | 0.033333 |
| compliance | 0.4 | 0.08 | 0.032 |
| crack_intensity | 0.6 | 0.3 | 0.18 |
| extent | 1.0 | 0.15 | 0.15 |
| element_damage | 0.5 | 0.15 | 0.075 |

## Report artifacts

- JSON: `C:\Users\ricmi\AppData\Local\Temp\aerotwin_eval_20260929T121713Z\outputs\reports\report_f2da23c6.json`
- PDF: `C:\Users\ricmi\AppData\Local\Temp\aerotwin_eval_20260929T121713Z\outputs\reports\report_f2da23c6.pdf`
- Size: 23979 bytes

## Notes

- Seeded run: random/NumPy/PyTorch RNGs fixed (seed=42) for reproducibility (domain rule 3).
- Pipeline ran in an isolated data dir: C:\Users\ricmi\AppData\Local\Temp\aerotwin_eval_20260929T121713Z.
- Model weights were NOT present in backend/app/models/weights/; deterministic random-init fixtures were used. Inference runtimes, dedup ratios, and scoring mechanics are valid, but task-level accuracy (IoU/F1/score calibration) would NOT be.
