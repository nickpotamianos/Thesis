Experiment cells in `thesis_run_all.ipynb`

This document documents only the notebook cells that actually launch, replay, or evaluate experiments (i.e., the cells that invoke the command-line runner `swarm_target_tracking.py`, spawn background runs, or call the evaluation wrapper `swarm_eval_table.py`). For each cell I list the cell number (as displayed in the notebook outline), its purpose, the exact command templates it uses, inputs and outputs, how failure is handled, and implementation notes you will find helpful when debugging or re-running experiments.

Important: The notebook defines a set of helper utilities used by these experiment cells. When you read the cell descriptions below keep in mind the following helpers are in the notebook and are reused by the experiment cells:

- `run(cmd, cwd=None)`: replaces a literal `python` with the notebook's `miluv_env` python (if present) and calls `subprocess.run(cmd, check=True, cwd=cwd)`. Any non-zero exit raises `CalledProcessError` and is visible in the notebook output.
- `spawn(cmd, cwd=None)`: same replacement but starts the process in background via `subprocess.Popen`.
- `unique_dir(path)`: returns a non-existent directory path by appending `_v2`, `_v3`, ... if the base exists, used to avoid accidental overwrites.
- `best_flags()`: returns the standard list of CLI flags appended to every `swarm_target_tracking.py` invocation (sensor usage flags, online tuning flags, gating params, etc.).

Cells that directly conduct experiments
-------------------------------------

Cell 17 — Cross‑Validation model training and evaluation

Purpose
- Train the per-fold models (BiasNet and FusionNet) and then evaluate every scenario on the held‑out test experiment for that fold.

What it does (high level)
- For each fold in the cross‑validation configuration:
  - Trains BiasNet and FusionNet variants (by_time, by_exp) using notebook helper functions.
  - Collects model directories for that fold.
  - Calls `eval_scenarios(...)` which replays the test experiment under multiple evaluation scenarios (Baseline, BiasNet-driven, FusionNet-driven, budgeted k, etc.).

How experiments are launched
- The cell does not directly construct raw subprocess strings in every spot; instead it calls `eval_scenarios(...)`. `eval_scenarios` constructs commands that look like:
  `[
     sys.executable, "swarm_target_tracking.py", "--exp", <experiment>, "--target", <target_id>, *best_flags(), "--out", <out_path>, <model-specific flags...> 
   ]`
- `eval_scenarios` uses `run(cmd, cwd=config['root'])` so the notebook's `run()` wrapper ensures the virtualenv python is used.

Inputs
- `fold_ds_dir`: path to the fold's dataset (built earlier in the notebook).
- `config['target']`: the target robot id (e.g., `ifo003`).
- Trained model directories returned by `train_biasnet()` and `train_fusionnet()`.

Outputs
- Per-scenario output directories under `paths['cv_root']` (one directory per fold / scenario). Each output directory is a unique directory (via `unique_dir`) and typically contains a run-level `all_summary.csv` produced by `swarm_eval_table.py`.
- Training artifacts under `paths['models_root']`.

Failure modes and notes
- Because `run()` uses `check=True`, any failing `swarm_target_tracking.py` invocation raises `subprocess.CalledProcessError` and is printed in the cell output. The notebook continues to the next fold when the exceptions are caught in loops or will stop entirely if not handled.
- The cell relies on the presence of the dataset cache (see `paths['cache_root']`); missing cache entries cause the fold to be skipped.

Why this cell is experimental
- It triggers the bulk of the runtime experiments: training and repeating the simulator / estimator over many scenarios and writing per-run summaries used by downstream analysis.


Cell 19 / Cell 20 / Cell 21 — Pattern‑based and alternate CV evaluation runners

Purpose
- These nearby code cells implement and run variations of the cross‑validation pipeline. They produce alternate fold definitions (pattern‑based grouping) or alternate evaluation groupings and then replay the experiments using the same runner.

What they do (high level)
- Build alternative fold lists (for example: group by experiment pattern like `default_3_random2_*` and exclude entire groups from training when testing any member).
- Reconstruct datasets for these pattern‑based folds (writing `bias_samples.jsonl` and `fusion_snaps.jsonl` files into `paths['ds_root']`).
- For each pattern fold they call the same `eval_scenarios(...)`/`run(cmd, ...)` approach to launch experiments on the held‑out test entries.

How experiments are launched
- Command templates are the same as in Cell 17: `sys.executable, "swarm_target_tracking.py", ...` plus the `--out` path. `unique_dir()` is used to avoid overwriting pre-existing results.

Inputs
- The folder of training experiments (cache entries) for the pattern-based training set.

Outputs
- Pattern fold dataset directories under `paths['ds_root']` and per-fold evaluation results under `paths['cv_root']`.

Failure modes and notes
- Missing cache directories for any training experiment cause the sample aggregation to skip that experiment and the fold may have fewer training samples than expected.
- As above, failing runs surface as CalledProcessError because `run()` uses `check=True`.


Cell cluster in the CV section that emits `swarm_target_tracking` invocations

(Cells in the 23..31 range contain the actual `cmd = [... "swarm_target_tracking.py" ...]` templates used repeatedly)

Purpose
- This group of cells defines the command templates and scenario-specific overrides used when the notebook replays experiments for all CV folds and evaluation regimes.

Key implementation details you should know
- The template always uses `sys.executable` as the interpreter entry in the command list; the notebook `run()` wrapper replaces `python` with the `miluv_env` python if a literal `python` was used elsewhere.
- Standard evaluation flags are centrally generated by `best_flags()` and then extended with scenario-specific options (for example, `--ci_method learned` and `--fusionnet_dir` when evaluating FusionNet-driven fusion).
- Each invocation sets an `--out <out_dir>` where `<out_dir>` is created via `unique_dir(...)` and therefore never silently overwrites previous runs.
- For budgeted experiments the notebook appends `--budget_k` or related flags to control the number of active tracker robots.

Why these cells are experimental
- They are the place where scenario-by-scenario experiment invocations are assembled — essentially the runbook for every experiment the notebook spawns during CV.


Cells that run the default_3_random experiments (single- and two-tracker baselines)
--------------------------------------------------------------------------

Cell 50 — Default_3_random runners (single- and two-tracker wrappers and orchestration)

Purpose
- This section orchestrates replaying the `default_3_random*` experiment family for two special baseline comparisons:
  - A single‑tracker baseline (force the estimator to observe only one tracker robot at a time)
  - A two‑tracker baseline (use both trackers available)

What this cell contains
- It discovers all experiments in `data/three_robots` with names starting with `default_3_random`.
- It reads `config['trackers']` (the notebook-level trackers list, e.g. `['ifo001', 'ifo002']`) as the set of tracker robots to test as single-tracker runs.
- It creates a run directory `single_tracker_root` under the main `runs` root (unique via `unique_dir`).

Single‑tracker runner (inner function)
- The cell defines `run_baseline_single_tracker(exp: str, tracker_id: str)` which:
  - Creates a unique `out_dir` for the replay (`single_tracker_root / tracker_id / exp`).
  - Builds the exact command list:
    `[
       sys.executable,
       "swarm_target_tracking.py",
       "--exp", exp,
       "--target", config['target'],
       *best_flags(),
       "--robots", tracker_id,
       "--out", str(out_dir)
     ]`
  - Calls `run(cmd, cwd=config['root'])` to execute the run.
  - Immediately after returns from the CLI run it calls `run([sys.executable, "swarm_eval_table.py", "--root", str(out_dir)], cwd=config['root'])` to produce `all_summary.csv` inside the same out directory.
  - If the summary file `all_summary.csv` is missing the function prints a warning and returns `None`. Otherwise it reads the summary into a DataFrame, annotates it with the experiment and tracker id, and returns it.

Two‑tracker runner (inner function)
- In the same code region the cell defines `run_baseline_two_tracker(exp: str)` which:
  - Joins the two tracker ids into `"ifo001,ifo002"` and uses the same invocation template as above but passes `"--robots", tracker_combo_label` (comma-delimited list).
  - Calls `swarm_target_tracking.py` via `run()` and then runs `swarm_eval_table.py` to produce `all_summary.csv`.
  - Converts the resulting `all_summary.csv` to a DataFrame annotated with `condition='two_robot'` and returns it.

Aggregation and output
- The cell loops over all discovered `default_3_random*` experiments and runs the single-tracker runner for each configured tracker, collecting individual DataFrames returned by each successful run.
- If no runs succeed the cell raises `RuntimeError("Single-tracker baseline runs produced no results; check logs above.")` which is what you observed during previous debugging.
- Successful runs are concatenated and written to `single_tracker_root / default_single_tracker_random_summary.csv`.

Failure modes and notes
- The notebook checks explicitly for `all_summary.csv` after the CLI run; missing summary leads to skipping that run (printing a warning). This commonly happens if the underlying `swarm_target_tracking.py` failed early, if the whitelist logic filtered out all robots (so nothing to evaluate), or if file permissions/paths prevented writing.
- Since `run()` uses `check=True`, a non-zero return from `swarm_target_tracking.py` will raise `CalledProcessError` unless it is caught by the calling loop. The notebook catches `subprocess.CalledProcessError` around individual run calls and continues to the next experiment.
- If you see the runtime error "Single-tracker baseline runs produced no results" then either every run returned without writing `all_summary.csv` or they all raised exceptions. Check the notebook cell stdout for the printed `[RUN] ...` commands and the CLI stdout/stderr that follow.

Why this cell is experimental
- This is an explicit ablation: it forces the estimator to see only a subset of robots (via `--robots`) and replays the same experiment logs with only that robot available. It is a canonical experiment cell used to produce the ablation results.


Cell 51 — Two‑robot baseline runner (paired with Cell 50)

Purpose
- Run the same `default_3_random*` family while using both trackers together (the canonical multi‑robot baseline). The cell is implemented next to the single‑tracker code and shares the same logic and output structure.

What it does
- Builds `two_tracker_root` under the notebook `runs` root using `unique_dir()`.
- For each `default_3_random*` experiment it constructs a command similar to the single‑tracker command but passes `--robots` a comma-separated list of the two trackers, e.g. `"ifo001,ifo002"`.
- Calls `run(...)` then `swarm_eval_table.py` and collects `all_summary.csv` for aggregation.
- Aggregates successful runs and writes `default_two_tracker_random_summary.csv`.

Failure modes and notes
- Same as single‑tracker: missing all_summary.csv indicates the per-run pipeline didn't finish producing a summary for that experiment.


Other cells that spawn or monitor experiments
-------------------------------------------

- Any cell that calls `spawn(cmd)` is starting a background process (logger nodes or the ROS loggers used during some experiments). Those cells are used when the notebook starts auxiliary processes; they do spawn runtime components but are not the primary experiment launches described above. Search the notebook for `spawn(` to find these helper spawners.

- Cells that call `swarm_eval_table.py` explicitly after runs (e.g., the `run([sys.executable, "swarm_eval_table.py", "--root", out_dir])` calls) are considered part of the experiment cell because they finalize the per‑run summary (`all_summary.csv`) that the rest of the notebook consumes.

Quick troubleshooting checklist when an experiment cell produces no results
--------------------------------------------------------------------------
1. Confirm `paths['cache_root']` points to an existing dataset cache with the expected `*_ifo003` subdirectories. The notebook prints where it expects cached collections early on.
2. Inspect the cell stdout for the exact `[RUN]` command printed by the `run()` helper; copy the command and run it manually with the notebook's `miluv_env` python to reproduce.
3. If a `CalledProcessError` occurred, inspect the CLI stdout/stderr printed in the same notebook cell to determine where `swarm_target_tracking.py` failed.
4. If the CLI succeeded but `all_summary.csv` is missing, inspect the run output directory for partial files (logs, snapshots) and check `swarm_target_tracking.py`'s internal logging — the per-run script may have early return conditions (e.g., no matched trackers after filtering) that cause it not to write a summary.
5. Make sure the EKF/estimator accepts the `--robots` whitelist (post-refactor requires both the CLI to pass the flag and the EKF to honour the whitelist in its state creation). If the CLI passed `--robots` but the estimator ignored it then the run might still succeed but not run the intended ablation.

Data and sensor assumptions for experiment replays
- Overview: When the notebook replays an experiment it consumes the previously collected sensor logs (the cache/dataset entries) and constructs measurement streams for the estimator and evaluation routines. The following assumptions are applied consistently by the runner and by the EKF measurement adapters so that runs are comparable across conditions:

- Tracker robots (e.g., `ifo001`, `ifo002`) — full-sensor assumption:
  - We assume trackers recorded and provide all typical sensor channels used by the estimator: UWB ranges (to anchors and to other robots), IMU (accelerometer/gyro), and local height/altitude measurements (laser/sonar/height sensor or tag‑based height).
  - Trackers are treated as fully instrumented nodes: inter-tracker UWB and tracker↔anchor communications are available and used by the tracker localization components and for relative measurements in the multi-robot EKF.
  - Any additional tracker-side measurements present in the cached logs (e.g., tag detections, ground-truth telemetry) are ignored unless the runner or CLI flags explicitly enable them.

- Target UAV (e.g., `ifo003`) — exact handling in this notebook (code‑referenced)

The notebook and runner implement the following concrete behavior for the target in the actual replay code (search for these names in `swarm_target_tracking.py` / merged source):

- Data ingestion (always queried): the devkit DataLoader is constructed with `imu='px4'` unconditionally and `height=True` only if `args.use_height or args.use_height_tf`. See `miluv = DataLoader(..., imu='px4', height=(args.use_height or args.use_height_tf))` and the subsequent variables `uwb_range`, `height_df`, and `imu_at_q` (variables defined in the runner).

- UWB ranges (always used for target updates): `uwb_range = _concat_with_robot(data, 'uwb_range')` is filtered to the active robots (the `robots` list built from requested robots + target). The per-tracker target filters and the EKF correction step both consume these aggregated UWB pairs: the EKF calls `ekf.correct({... 'range': ..., 'to_id': ..., 'from_id': ...})` and the per-tracker `TargetIF` receives the robust-aggregated range `z_agg` and uses `target_filters[trk].correct(z_corr, R_eff, tracker_pos=eff_sensor_pos)` to update local posteriors.

- Height measurements (used only when enabled): `height_df` is created only when `(args.use_height or args.use_height_tf)` is true. Time-aligned height series `height_at_q` are built only when `args.use_height_tf` is true and non-empty. EKF height corrections are applied when `args.use_height` is true (`ekf.correct({'height': ..., 'robot': ...})`). The per-tracker filters use height-difference corrections only when `args.use_height_tf` and `height_at_q` are available (`target_filters[trk].correct_height(dz_meas, z_trk, R_h)`). ML feature construction includes `height_tracker` and `height_target` entries only when `args.use_height_tf` is set and `height_at_q` exists.

- IMU usage (precise): IMU data are queried for every active robot (`imu_at_q = miluv.query_by_timestamps(..., sensors='imu_px4')`) and are used by the authors' multi-robot EKF predict step: the code builds `u_dict` from `gyro` and `accel` and calls `ekf.predict(u_dict, dt)`. Thus the target's IMU (if present in the cache and the target is active in `robots`) is used by the EKF state propagation. However, the per-tracker TargetIF filters and the ML features do not consume the target's IMU signals — their updates are driven by aggregated UWB ranges and optional height differences.

- Inclusion guarantee: the runner enforces that the CLI `--target` (or notebook `config['target']`) is included in the active robot set; the code raises `ValueError` if the target is absent after requested-robot filtering. Search for the runtime check that raises "Target '<id>' must be included in active robot set".

Consequences and reproducibility

- Because `best_flags()` in the notebook includes both `--use_height` and `--use_height_tf` by default, the typical notebook evaluation runs will enable height processing and thereby include height measurements and height-difference corrections in features and target-filter corrections. If you want to run with target heights disabled, remove those flags from the invocation for that experiment.

- Summary (concise test-time facts):
  - UWB ranges from target↔tracker are always used to update both the EKF and the per-tracker TargetIFs.
  - Height is used when height flags are enabled; time-aligned height (use_height_tf) enables target↔tracker dz corrections and inclusion in ML features.
  - Target IMU is loaded and used by the EKF predict step but is not an input to the per-tracker target filters or ML feature vectors in the current replay pipeline.

This section replaces the earlier heuristic statement and maps it to the exact variables and functions used by the notebook-runner code so you can verify or change behavior by editing the corresponding flags or code paths.

- Anchors and their roles:
  - UWB anchors are assumed to have known, fixed positions (this is required by the measurement models that convert anchor ranges into position constraints for tracker nodes).
  - Anchor-to-robot ranges in the cached logs are used by the tracker localization submodules; the target uses only ranges to trackers in the ablation condition (anchor-to-target ranges are normally absent or unused in this setup).

- How the runner enforces these assumptions
  - The notebook and CLI cooperate to filter datasets before replay: the `--robots` whitelist removes all measurements belonging to excluded robot IDs; measurement adapters inside `swarm_target_tracking.py` further select specific sensor channels (UWB, height, IMU) according to the experiment configuration and the explicit flags present for that run.
  - Precisely: the notebook's standard evaluation invocations call `best_flags()`; that helper list includes both `--use_height` and `--use_height_tf`, therefore the default experiment runs in this notebook enable height processing. The code constructs height dataframes only when `args.use_height` or `args.use_height_tf` is true (implementation: `height_df = _concat_with_robot(data, "height") if (args.use_height or args.use_height_tf) else pd.DataFrame(columns=["timestamp"])`). When height is enabled the pipeline will include height measurements both in per-tracker target filters (TargetIF) and in BiasNet/FusionNet feature construction; when `--use_height_tf` is set the notebook additionally uses height time-aligned transforms (delta-z features) where applicable.
  - The target filter and per-tracker target predictors use flags from the runner to configure process noise and gating (for example `--sigma_a_xy`, `--sigma_a_z`, `--gate_sigma_init`, `--gate_target`); these parameters are supplied by `best_flags()` in the notebook, so they are consistently applied across experiments unless explicitly overridden.
  - After filtering, the EKF is constructed to reflect only the active robots (the whitelist), and measurements are supplied only from the selected channels.
  - The evaluation pipeline (calls to `swarm_eval_table.py`) expects the same canonical `all_summary.csv` fields regardless of which sensors were active, so downstream analyses remain comparable.

- Why these assumptions matter for comparisons
  - Apples-to-apples: To meaningfully compare single-tracker vs two-tracker conditions the only difference should be the set of active robots and therefore the source of range information. By fixing the target to only provide UWB+height and treating trackers as fully instrumented, the experiments isolate how additional tracker measurements (peer UWB, anchor ranges, IMU-assisted tracker localization) influence target localization performance.
  - Underconstrained geometry: With only a single tracker and target↔tracker ranges, the estimator has fewer independent spatial constraints; the added height measurement (and motion model priors) partially relieve this, but RMSE differences should be interpreted in light of these geometric observability limitations.

+Decentralized experiments (where and what they use)
+- Location in the notebook: Section "UDP Decentralized Demos" and the subsequent "Full Decentralized Multi-UAV Evaluation" (these are the decentralized evaluation cells that enumerate DECENTRALIZED BASELINE / DECENTRALIZED ML runs across pattern folds).
+- What changes vs centralized runs:
+  - Data channels: decentralized runs use the same local sensor logs as centralized runs (UWB ranges, IMU, height, anchor ranges) for each robot — the local observations are not changed. The distinction is in the fusion strategy, not the raw measurements.
+  - Communication model: when `--decentralized` is enabled the runner engages a gossip-based CI (covariance intersection) exchange protocol: communication parameters such as `--comm_p` (per-link probability), `--comm_drop` (drop probability), `--comm_rounds` (number of gossip rounds), and `--comm_seed` are used to simulate lossy, probabilistic message passing between robots. The notebook's decentralized runs explicitly set or default those parameters within the decentralized experiment cells so that results are reproducible.
+  - Local processing: each robot runs its local target filter (TargetIF) using its own measurements. After local updates, the decentralized fusion step exchanges information (in information form) over the simulated network and fuses per-robot posteriors via the gossip CI fuser. This means the per-robot data inputs are identical to centralized runs, but the fused estimate results differ because fusion is distributed and communication is imperfect.
+  - Why this matters: decentralized tests therefore measure the impact of communication constraints and fusion strategy on final RMSE while holding sensing inputs constant.

- Checklist to verify data assumptions before running
- 1. Confirm the cached dataset (paths['cache_root'] / <robot> / <exp>_ifo003 or similar) contains the expected files and channels for each robot.
- 2. For tracker nodes ensure UWB, IMU and height channels exist in the cache; for target check that UWB ranges to trackers and a height channel exist (or plan to filter the cache to keep only these).
- 3. Verify experiment configuration flags: by default the notebook's evaluation functions append `best_flags()` which includes the height and tuning flags (`--use_height`, `--use_height_tf`, `--uwb_std`, `--pair_corr`, `--sigma_a_xy`, `--sigma_a_z`, `--gate_target`, `--gate_sigma_init`, `--q_adapt`, `--r_floor_blend`, etc.). If you need a different data usage policy for a particular experiment (for example, explicitly disable height or enable decentralized gossip), set or remove the relevant flags in that cell's invocation rather than editing data files directly.
- 4. If anchor positions are not correct or missing, provide the correct anchor map to the runner (the runner/EKF requires anchors' positions to transform ranges into constraints).
- 5. Inspect a sample run output directory after a single replay to ensure `swarm_eval_table.py` produced `all_summary.csv` with expected fields (rmse_3d, per-robot stats, etc.).

Final note
- This explanation deliberately documents only the cells that launch or summarize experiments. Many other notebook cells prepare datasets, define plotting helpers, or analyze aggregated CSV files — those are intentionally excluded here.

If you want, I can also:
- Add a concise mapping table listing the exact cell numbers (from the currently loaded notebook) and the first line of code for each experimental cell for quick navigation, or
- Add a short script snippet you can copy/paste to re-run a problem run from the command-line for faster debugging.

Definitive sensor inputs used by the notebook runs (no conditions)
- Context: The notebook constructs and calls `swarm_target_tracking.py` with `*best_flags()` for every evaluation command in the CV and ablation cells. `best_flags()` in the notebook explicitly includes `--use_height` and `--use_height_tf`, plus the tuning and sensor flags listed below. Therefore the statements that follow describe the exact data channels the notebook uses when it runs experiments as committed.

- CLI flags included by the notebook's `best_flags()` (exact list used in invocations):
  --use_height, --use_height_tf, --uwb_std 0.8, --pair_corr 0.3, --sigma_a_xy 3.0, --sigma_a_z 1.5, --ci_method grid, --ci_objective trace, --los_influence 0, --geom_influence 0, --ema_alpha 0.0, --online_tune, --online_r_min_scale 0.75, --online_r_max_scale 3.0, --gate_target 0.90, --gate_sigma_init 4.0, --q_adapt, --r_floor_blend 0.5

- Exact sensor channels used for trackers (as executed by the notebook):
  - UWB ranges (inter-tracker and tracker↔anchor): collected into the variable `uwb_range = _concat_with_robot(data, "uwb_range")` and filtered to the active robots. These ranges are used by:
    - the authors' EKF correction step via `ekf.correct({"range": ..., "to_id": ..., "from_id": ...})`;
    - the per-tracker target filters (robust aggregation → `z_agg`) and later `target_filters[trk].correct(z_corr, R_eff, tracker_pos=...)`.
  - IMU (px4): queried as `imu_at_q = miluv.query_by_timestamps(..., sensors='imu_px4')` and split into `gyro` and `accel` structures. The EKF uses these signals in the predict step via `u_dict` and `ekf.predict(u_dict, dt)`.
  - Height/altitude: the notebook creates `height_df = _concat_with_robot(data, "height")` and `height_at_q` (time-aligned height array). Tracker height entries are used:
    - by the EKF for explicit height corrections `ekf.correct({"height": ..., "robot": ...})`;
    - by per-tracker filters for z-difference corrections via `target_filters[trk].correct_height(dz_meas, z_trk, R_h)`;
    - included in ML feature vectors as `height_tracker` when the height time-series is present.
  - Anchors: anchor map `miluv.anchors` is loaded and passed to the EKF to interpret anchor→robot ranges; anchor ranges are used in tracker localization computations.

- Exact sensor channels used for the target (as executed by the notebook):
  - UWB ranges (target↔tracker pairs): sourced from the same `uwb_range` aggregation and used by:
    - the authors' EKF correction (`ekf.correct(...)`) which consumes any range messages that include the target;
    - the per-tracker TargetIF update flow where robust-aggregated `z_agg` (target↔tracker) is used in `target_filters[trk].correct(...)`.
  - Height/altitude: the notebook constructs `height_df` and `height_at_q` and uses the target's height values in:
    - EKF height corrections (`ekf.correct({'height':..., 'robot':<target>})`);
    - per-tracker height-difference corrections `target_filters[trk].correct_height(dz_meas, z_trk, R_h)`;
    - ML feature vectors as `height_target` when time-aligned height is present.
  - IMU (px4): the target's IMU is included in `imu_at_q` and the built `u_dict` is passed to `ekf.predict(u_dict, dt)`. Thus the EKF state propagation uses the target's IMU data (if present in the cached data and the target is active in the `robots` set). Note: per-tracker TargetIFs and ML features do not consume the target's IMU — only the EKF predict uses IMU for all active robots.

- Decentralized experiments: the decentralized evaluation cells use the same raw sensor channels for each robot (UWB, IMU, height, anchor ranges) as the centralized runs; the difference lies only in the fusion pathway (gossip CI) and the communication simulation parameters supplied to the runner.

- Summary (one-line, definitive): In the notebook's committed experiment invocations the pipeline uses — for trackers: UWB ranges (to anchors & peers), IMU (px4 gyro+accel), height/altitude, and anchor maps; for the target: UWB ranges to trackers, height/altitude, and the target's IMU used in EKF.predict. The per-tracker TargetIF updates and ML feature vectors use only UWB ranges and height-derived values (height differences / time-aligned heights) and do not consume the target IMU directly.
