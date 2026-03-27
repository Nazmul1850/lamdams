# lamdams — PBC Pipeline

End-to-end pipeline for compiling and scheduling Pauli-Based Computation (PBC) circuits onto a hardware graph.

## Entry point

```
python run_compile.py [options]
```

The `META` dict in `run_compile.py` controls every knob. All keys are optional — unrecognised keys are silently ignored. Any key can also be overridden from the command line without touching the source file.

---

## Pipeline modes

| `compiled` | Input | What happens |
|---|---|---|
| `True` (default) | Pre-compiled `PBC.json` | Skips the QASM frontend, runs stages 2–8 directly |
| `False` | `.qasm` circuit file | Runs the full pipeline: QASM → PBC.json (stage 1) → stages 2–8 |

---

## META reference

### Hardware

| Key | Type | Default | Description |
|---|---|---|---|
| `topology` | `"grid"` \| `"ring"` | `"grid"` | Hardware graph topology |
| `sparse_pct` | float [0, 1) | `0.0` | Fraction of qubit slots left empty; `0.0` = fully dense |
| `n_data` | int | `11` | Data qubit slots per block (Gross code = 11) |
| `coupler_capacity` | int | `1` | Maximum parallel operations per coupler link |

### Mapping

| Key | Type | Default | Description |
|---|---|---|---|
| `mapper` | str | `"auto_round_robin_mapping"` | Logical-to-physical qubit mapper |
| `sa_steps` | int | `10000` | Simulated-annealing iterations (SA mapper only) |
| `sa_t0` | float | `1e5` | SA start temperature |
| `sa_tend` | float | `1.1` | SA end temperature |

Available mappers: `auto_round_robin_mapping`, `pure_random`, `simulated_annealing`, `auto_pack`, `random_pack_mapping`

### Scheduling

| Key | Type | Default | Description |
|---|---|---|---|
| `scheduler` | str | `"greedy_critical"` | Operation scheduler |
| `cp_sat_time_limit` | float | `120.0` | Per-layer time budget in seconds (CP-SAT scheduler only) |

Available schedulers: `sequential_scheduler`, `greedy_critical`, `cp_sat`

### Pipeline control

| Key | Type | Default | Description |
|---|---|---|---|
| `compiled` | bool | `True` | `True` = start from `PBC.json`; `False` = start from QASM |
| `compact` | bool | `True` | Save PBC in compact format (`True`) or full cache format (`False`) |
| `seed` | int | `42` | Global RNG seed |

### Experiment flags (Fig 8 / 9 / 10)

| Key | Type | Default | Description |
|---|---|---|---|
| `run_experiments` | bool | `True` | Run multi-mapper/scheduler comparison experiments; set `False` to skip and save time |
| `exp_sparse_pct` | float | `0.7` | Sparsity used in the sparse-vs-dense comparison (Fig 9) |
| `exp_mapper` | str | `"simulated_annealing"` | Mapper used in the sparse-vs-dense run |
| `exp_scheduler` | str | `"greedy_critical"` | Scheduler used in the sparse-vs-dense run |

---

## Switching circuits

Edit `ACTIVE_RUN` in `run_compile.py` to pick from the `PBC_PATHS` dict, or pass `--pbc` on the command line:

```python
PBC_PATHS = {
    "randon_10_100t": "runs/randon_10_100t_v2__seed42__.../stage_frontend/PBC.json",
    "rand_50q_1kt":   "runs/rand_50q_1kt__seed42__.../stage_frontend/PBC.json",
    "test_rotations": "runs/test-rotations/PBC.json",
}

ACTIVE_RUN = "randon_10_100t"   # ← change this to switch circuits
```

---

## Running examples

### Default run (compiled PBC, grid topology, greedy scheduler)

```bash
python run_compile.py
```

### Switch to ring topology with 50 % sparse hardware

```bash
python run_compile.py --topology ring --sparse_pct 0.5
```

### Use the CP-SAT scheduler with a 60-second per-layer budget

```bash
python run_compile.py --scheduler cp_sat --cp_sat_time_limit 60
```

### Use simulated-annealing mapper, skip experiments

```bash
python run_compile.py --mapper simulated_annealing --run_experiments false
```

### Point at a specific PBC file directly

```bash
python run_compile.py --pbc path/to/my/PBC.json
```

### Full QASM run (compile from source, then run the pipeline)

```bash
python run_compile.py --compiled false --pbc path/to/circuit.qasm
```

### Reproduce Fig 9 sparse-vs-dense comparison with custom settings

```bash
python run_compile.py \
    --exp_mapper simulated_annealing \
    --exp_scheduler greedy_critical \
    --exp_sparse_pct 0.6 \
    --seed 7
```

---

## Output

Each run creates a timestamped directory under `runs/` containing:

```
runs/<tag>__seed<N>__<timestamp>/
    config.json                  pipeline config snapshot
    trace.ndjson                 event log (one JSON object per line)
    stage_frontend/
        PBC.json                 compiled Pauli-Based Computation circuit
    figures/
        fig_01_circuit_character.png
        fig_02_depth_profile.png
        fig_03_block_utilization.png
        fig_04_routing_distances.png
        fig_05_frame_rewrites.png
        fig_06_parallelism_profile.png
        fig_08_algo_comparison.png
        fig_09_sparse_dense.png
        fig_10_topology_gallery.png
```
