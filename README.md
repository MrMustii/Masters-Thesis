# Regional GenCast

Master's thesis code (DTU). Adapts DeepMind's GenCast from the global 0.25 degree grid to a
limited area European domain (45 to 65 N, 5 W to 25 E, plus an 8.5 degree boundary margin) and
evaluates it on 2m temperature, daily maxima, the diurnal cycle and heatwaves. Trained on ERA5
2005 to 2019, validated on 2020, evaluated on 2021. Runs on LUMI.

## Requirements

Everything runs inside the Singularity image at `containers/my_new_container2.sif`, built from
`containers/update.def` on top of an image described by `containers/conda_venv.yml`. The
environment is ROCm specific because LUMI's nodes are AMD, so a plain `pip install` will not
reproduce it.

Input data lives on LUMI-O object storage. Credentials are needed in two places, in different
formats, and nothing works until both are present. `~/.lumio_creds` is sourced by bash and needs
`export LUMIO_ACCESS_KEY=...` and `export LUMIO_SECRET_KEY=...`. `~/.config/rclone/rclone.conf`
is INI and needs `access_key_id` and `secret_access_key` under each remote section. Keys expire;
regenerate them with `lumio-conf` and update both files.

## Repository layout

`modified_Gencast/` is the thesis code: the LUMI-O data layer and samplers, the model and noise
constructions, the training loop, four inference entry points, the metrics and significance
libraries, the per experiment reporting scripts and the comparison scripts.

`job_scripts/` holds the Slurm launchers that drive the pipeline. Four of them target LSF from
an earlier phase on DTU's cluster and are not part of the current pipeline: `bsub.sh`,
`DTU_download.sh`, `DTU_hpc.sh` and `scheduler.sh`.

`Googles_gencast/` is DeepMind's code, vendored unmodified. Do not edit it.

`containers/` holds the Singularity definition and the conda environment.

`gencast_infernce/` is the initial prototype, kept as a record of that stage. Its `config.py`
is still live and resolves the cluster paths every other module imports.

Trained checkpoints and evaluation output are on LUMI scratch, input data is on LUMI-O, and
training curves are on Weights and Biases under the project `Gencast`. None of it is in this
repository.

## Running the pipeline

Five stages, each depending only on the previous one's output. All are Slurm jobs.

```bash
sbatch job_scripts/lumi_build.sh             # build the ERA5, climatology and threshold stores
sbatch job_scripts/lumi_train.sh             # train one checkpoint (two stages)
sbatch job_scripts/lumi_eval.sh              # roll a checkpoint forward, cache predictions
sbatch job_scripts/lumi_report_exp2.sh       # predictions to one metric cache per run
sbatch job_scripts/lumi_compare.sh           # metric caches to figures and CSVs
```

`lumi_build.sh`, `lumi_train.sh` and `lumi_eval.sh` pick a pipeline from a `MODE` environment
variable; read the header of each for the options. Reporting has one launcher per experiment,
`lumi_report_exp1.sh` through `lumi_report_exp4.sh`, plus `lumi_report_baselines.sh`. The
baseline launcher must run before any comparison, because every skill score needs its
denominators.

Comparisons are modules rather than job scripts:

```bash
python -m modified_Gencast.comparison_23
```

Predictions are written on the full loading grid, including the boundary margin. The reporting
scripts crop the 8.5 degree belt themselves, so do not pre-crop the files: four of the five
shrink whatever grid they are given, and a cropped file would be cropped twice.

## Experiments

Config files are in `modified_Gencast/traninig configs/`, one per run.

**exp1** runs Google's pretrained checkpoint over the region without retraining. Entry point
`eval_original_weights.py`. It reads the `-tp12` data store, because the pretrained checkpoint
expects precipitation that the WeatherBench2 ERA5 archive does not carry.

**exp2** is seven models differing only in input variables: `surface`, `temp1000`, `temp850`,
`z500`, `z700`, `combined`, `all_gencast`. Fixed 12 hour steps. Entry point `eval.py`, which
also takes `--boundary-margin-degrees` for the margin ablation.

**exp3** is three lead time conditioned models at horizons of 24, 72 and 168 hours, the only
difference between their configs, rolled forward one hour at a time. Entry point
`eval_continuous.py`, with `--noise-method` selecting fixed or Ornstein-Uhlenbeck noise and
`--ou-rho` setting the correlation length.

**exp4** reconstructs a day's 24 hourly fields from its daily mean, maximum and minimum. Four
models vary `day_offsets`: `[0]`, `[-1,0,1]`, `[-2,-1,0]`, `[-2,-1,0,1,2]`. Entry point
`eval_daily.py`. Its `--min-lat`, `--max-lat`, `--min-lon` and `--max-lon` flags point a trained
checkpoint at a different region without retraining, which is how the NW America and N Africa
runs were produced.

## Comparisons

Eight comparisons across five scripts. The comparison numbers do not line up with the experiment
numbers.

| script | comparisons | reads |
|---|---|---|
| `comparison_0.py` | 0 | the baseline caches |
| `comparison_1.py` | 1 | exp1 and exp2 caches |
| `comparison_23.py` | 2, 3 | exp2 caches |
| `comparison_45.py` | 4, 5 | exp3 caches |
| `comparison_67.py` | 6, 7 | exp4 caches |

Each reads only the metric caches, so re-analysing a result never requires re-running a model.
Each writes `primary_scores.csv`, the paired and pairwise significance tables, and
`diagnostics.csv` into its own output directory.

## Attribution

Built on GenCast and GraphCast by Google DeepMind, included under the Apache License 2.0 in
`Googles_gencast/` at upstream commit `c037656`. Upstream has since renamed that package, so
paths here will not match current upstream.

Four files in `modified_Gencast/` are derived from it: `gencast.py`, `denoiser.py`, `losses.py`
and `regional_utils.py`. Each carries DeepMind's copyright notice, names the upstream file it
descends from, and states what changed. In `gencast.py` and `denoiser.py`, modified blocks are
delimited by `MODIFIED FROM UPSTREAM` markers, and anything outside them is DeepMind's. Files
carrying no third party notice are original to this work.

## References

GenCast: Price, I. et al. (2025). Probabilistic weather forecasting with machine learning.
*Nature*, 637, 84 to 90. <https://arxiv.org/abs/2312.15796>

GraphCast: Lam, R. et al. (2023). Learning skillful medium-range global weather forecasting.
*Science*, 382(6677), 1416 to 1421. <https://www.science.org/doi/10.1126/science.adi2336>

Code for both: <https://github.com/google-deepmind/graphcast>

WeatherBench 2: Rasp, S. et al. (2024). WeatherBench 2: a benchmark for the next generation of
data-driven global weather models. *Journal of Advances in Modeling Earth Systems*, 16(6).

Andrae et al. (2025). Source of the Ornstein-Uhlenbeck noise construction used in exp3; see
Algorithm 2, referenced in `modified_Gencast/ou_noise.py`.
