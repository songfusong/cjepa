# ECO-JEPA Implementation Progress

Last updated: 2026-06-16

## Summary

This repository currently contains an ECO-JEPA MVP built on top of the existing
C-JEPA PushT slot-latent pipeline. The implemented contribution is focused on
event-conditioned object-history masking and masking ablations for PushT.

This is not yet the full ECO-JEPA v2 plan. The current implementation does not
yet include a contact graph prediction head, relative transition loss,
multi-horizon event-centric rollout metrics, action-ranking correlation, or
PushCube / articulated-contact tasks.

## Implemented Components

### Masking strategies

`src/cjepa_predictor.py` now supports four masking strategies:

- `random`: C-JEPA-style random object-slot masking baseline.
- `event_window`: ECO-JEPA MVP masking using precomputed contact event windows.
- `contact`: masking conditioned directly on binary contact labels.
- `distance`: masking conditioned on contact distance thresholding.

The AP predictor still excludes the last two AP tokens from object masking
because those tokens correspond to proprioception and action context. This keeps
the current masking intervention aligned with object-level masking rather than
masking all tokens indiscriminately.

### Contact-event data loading

`src/custom_codes/custom_dataset.py` supports loading contact-event metadata via
`contact_event_dir` and `contact_event_fields`.

Useful field modes:

- `full`: loads contact, event mode, onset, persistent, release, post-contact,
  event window, and distance-related fields when available.
- `minimal`: loads only the fields needed by event/contact masking.
- `minimal,distance`: loads minimal fields plus `contact_distance` for the
  distance masking baseline.

### Training integration

`src/train/train_causalwm_AP_node_pusht_slot.py` now forwards `event_window`,
`contact`, and `contact_distance` from the dataset batch into the predictor.
The training config reads ECO-JEPA options from:

- `ecojepa.enabled`
- `ecojepa.masking`
- `ecojepa.p_base`
- `ecojepa.p_event`
- `ecojepa.min_masked_slots`
- `ecojepa.max_masked_slots`
- `ecojepa.distance_threshold`

### Scripts

The current PushT scripts are under `scripts/pusht/`:

- `train_pusht_masking_ablation.sh`: unified launcher for the four masking
  strategies.
- `eval_pusht_policy.sh`: runs event-centric prediction eval and/or planning
  eval for a named policy.
- `summarize_masking_ablation.py`: combines event prediction and planning
  results into one CSV.
- `print_masking_ablation_commands.sh`: prints reproducible training commands.

## Current Experiments

### Previous MVP sanity comparison

The previous C-JEPA vs ECO-JEPA MVP comparison was a sanity check, not a formal
experiment. It used one seed and earlier scripts, so it should not be treated as
final evidence.

Event-centric prediction files:

- `outputs/pusht_event_prediction_cjepa_paper_m1_val.json`
- `outputs/pusht_event_prediction_ecojepa_eventmask_m1_val.json`

Observed one-step slot MSE:

| Policy | Overall | Onset | Persistent | Release | Post-contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pusht_cjepa_paper_m1` | 0.002252 | 0.002342 | 0.002853 | 0.002270 | 0.002362 |
| `pusht_ecojepa_eventmask_m1` | 0.002241 | 0.002276 | 0.002825 | 0.002391 | 0.002288 |

Planning files:

- `src/plan/planning_pusht_cjepa_paper_m1_val50_seed0.txt`
- `src/plan/planning_pusht_ecojepa_eventmask_m1_val50_seed0.txt`

Observed planning success:

- `pusht_cjepa_paper_m1`: 46% on 50 PushT validation episodes.
- `pusht_ecojepa_eventmask_m1`: 52% on 50 PushT validation episodes.

Interpretation: this is early evidence that event-conditioned masking may help,
especially around onset and post-contact, but the comparison is not yet strong
enough for a paper claim.

### Current masking ablation run

The current fairer seed-0 masking ablation is running with:

- `seed=0`
- `batch_size=1024`
- `num_workers=0`
- `max_epochs=30`
- PushT slot-latent setup
- one masked object slot
- `frameskip=5`
- `history_size=3`
- `num_preds=1`

Current model names:

- `pusht_cjepa_random_m1_seed0_nw0`
- `pusht_ecojepa_eventwindow_m1_seed0_nw0`
- `pusht_ecojepa_contact_m1_seed0_nw0`
- `pusht_ecojepa_distance_m1_seed0_nw0`

The earlier attempt with `num_workers=2` stalled during validation with LanceDB
fork warnings. Logs were saved under:

- `outputs/pusht_training_logs/`

Those stalled runs should be treated as failed runs, not formal results.

## Reproducible Commands

Train one masking strategy:

```bash
cd /data/users/jiaoyihang/cjepa
source /data/users/jiaoyihang/miniconda3/etc/profile.d/conda.sh
conda activate cjepa

CUDA_VISIBLE_DEVICES=0 \
SEED=0 \
MASKING_STRATEGY=event_window \
BATCH_SIZE=1024 \
NUM_WORKERS=0 \
MAX_EPOCHS=30 \
bash scripts/pusht/train_pusht_masking_ablation.sh \
  output_model_name=pusht_ecojepa_eventwindow_m1_seed0_nw0 \
  wandb.enable=false
```

Run event-centric prediction eval:

```bash
POLICY=pusht_ecojepa_eventwindow_m1_seed0_nw0 \
RUN_PLANNING=0 \
bash scripts/pusht/eval_pusht_policy.sh
```

Run planning eval:

```bash
POLICY=pusht_ecojepa_eventwindow_m1_seed0_nw0 \
RUN_EVENT=0 \
NUM_EVAL=50 \
bash scripts/pusht/eval_pusht_policy.sh
```

Summarize the four seed-0 ablation policies:

```bash
python scripts/pusht/summarize_masking_ablation.py \
  --policies \
    'pusht_cjepa_random_m1_seed{seed}_nw0' \
    'pusht_ecojepa_eventwindow_m1_seed{seed}_nw0' \
    'pusht_ecojepa_contact_m1_seed{seed}_nw0' \
    'pusht_ecojepa_distance_m1_seed{seed}_nw0' \
  --seeds 0 \
  --num-eval 50 \
  --output-csv outputs/pusht_masking_ablation_summary_seed0_nw0.csv
```

## Current Limitations

- The current optimizer path still uses the repository's Lightning AdamW setup,
  not a verified exact Adam optimizer reproduction from the C-JEPA paper.
- The current ablation uses `batch_size=1024` because it was requested for the
  active run. The C-JEPA paper PushT setting was described as `batch_size=256`.
- The current event-centric eval is one-step prediction over slot embeddings. It
  is useful for early diagnosis, but it is not yet the final multi-horizon
  event-centric rollout evaluation from the ECO-JEPA v2 plan.
- Current planning eval is standard PushT CEM/MPC evaluation, but action-ranking
  correlation has not yet been implemented.
- PushCube, Drawer / articulated contact, OOD tests, noisy contact ablations,
  and influence analysis are still pending.

## Next Steps

1. Let the four seed-0 masking ablation models finish training.
2. Run event-centric prediction eval for all four policies.
3. Run planning eval for all four policies with the same seed and episode set.
4. Summarize random vs event-window vs contact vs distance.
5. Use the ablation result to decide whether event-window masking is stronger
   than contact-only and distance-only masking.
6. If the ablation is promising, implement the next ECO-JEPA components:
   contact graph prediction head, relative transition loss, multi-horizon
   rollout metrics, and action-ranking correlation.

## Evidence Level

Current evidence level: sanity / MVP evidence only.

The implementation is useful for deciding whether the ECO-JEPA masking idea is
worth continuing. It is not yet sufficient for a formal claim that ECO-JEPA is
better than C-JEPA.
