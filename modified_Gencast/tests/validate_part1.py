#!/usr/bin/env python3
"""Self-contained validation of the Part I shared-foundation metrics.

No LUMI-O, no stores — builds synthetic ensembles with known answers and checks
each new/changed function against them (perfect forecast → 0, deterministic
collapse, controlled tail width, controlled peak offset, etc.). Exits non-zero on
any failure so the SLURM log shows PASS/FAIL at a glance.

    python modified_Gencast/tests/validate_part1.py
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modified_Gencast.metrics import (
    crps_field, crps, variogram_score, spatial_variogram_score,
    diurnal_shape_errors, peak_time_ccrps,
    tail_ratio, conditional_rank_histogram, probabilistic_detection,
    over_smoothing_diagnostic,
)
from modified_Gencast.significance import equivalence_test

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def hourly_dataarray(values_ensemble_forecast_hour, lat, lon):
    """Wrap a (ensemble, forecast, hour) array into an (…, lat, lon) DataArray
    with every cell identical, hour coord 0..H-1."""
    ensemble, forecast, hours = values_ensemble_forecast_hour.shape
    broadcast = np.broadcast_to(
        values_ensemble_forecast_hour[..., None, None], (ensemble, forecast, hours, len(lat), len(lon)))
    return xr.DataArray(np.array(broadcast), dims=["ensemble", "forecast", "hour", "lat", "lon"],
                        coords={"ensemble": np.arange(ensemble), "forecast": np.arange(forecast),
                                "hour": np.arange(hours), "lat": lat, "lon": lon})


def main():
    rng = np.random.default_rng(0)
    lat = np.array([54.0, 55.0])
    lon = np.array([10.0, 11.0])
    hours = np.arange(24)
    diurnal = 285.0 + 6.0 * np.sin(2 * np.pi * (hours - 9) / 24)  # peaks ~ hour 15

    # ── CRPS family ──────────────────────────────────────────────────────────
    print("CRPS / twCRPS")
    target = hourly_dataarray(np.tile(diurnal, (1, 3, 1)), lat, lon).isel(ensemble=0, drop=True)
    perfect = hourly_dataarray(np.tile(diurnal, (8, 3, 1)), lat, lon)
    check("fair CRPS perfect ≈ 0", abs(float(crps(perfect, target)["crps"].mean())) < 1e-9)
    deterministic = perfect.isel(ensemble=0, drop=True) + 2.0
    check("CRPS deterministic == |x-y|",
          abs(float(crps(deterministic, target)["crps"].mean()) - 2.0) < 1e-9)
    below_tau = target - 50.0                      # everything under a high τ
    tau = xr.full_like(target, 310.0)
    check("twCRPS all-below-τ ≈ 0",
          abs(float(crps(below_tau, below_tau, threshold=tau)["crps"].mean())) < 1e-9)

    # ── Variogram (temporal + spatial) ───────────────────────────────────────
    print("Variogram")
    check("temporal VS perfect ≈ 0", abs(variogram_score(perfect, target)["scalar"]) < 1e-9)
    check("temporal VS non-negative", variogram_score(perfect + rng.normal(0, 1, perfect.shape), target)["scalar"] >= 0)
    grid_lat = np.linspace(54, 58, 6)
    grid_lon = np.linspace(8, 12, 6)
    field = 285 + 3 * np.sin(np.linspace(0, 3, 6))[:, None] + 2 * np.cos(np.linspace(0, 3, 6))[None, :]
    spatial_target = xr.DataArray(np.broadcast_to(field, (2, 24, 6, 6)),
                                  dims=["forecast", "hour", "lat", "lon"],
                                  coords={"forecast": [0, 1], "hour": hours, "lat": grid_lat, "lon": grid_lon})
    spatial_perfect = spatial_target.expand_dims(ensemble=[0, 1])
    check("spatial VS perfect ≈ 0", abs(spatial_variogram_score(spatial_perfect, spatial_target)["scalar"]) < 1e-9)
    check("spatial VS non-negative",
          spatial_variogram_score(spatial_perfect + rng.normal(0, 1, spatial_perfect.shape), spatial_target)["scalar"] >= 0)

    # ── Transformed proper scores ────────────────────────────────────────────
    print("DTR / ASY / PTE")
    diurnal_shape = diurnal_shape_errors(perfect, target)
    check("DTR-CRPS perfect ≈ 0", abs(diurnal_shape["dtr_crps_scalar"]) < 1e-9)
    check("ASY-CRPS perfect ≈ 0", abs(diurnal_shape["asymmetry_crps_scalar"]) < 1e-9)
    check("PTE-cCRPS perfect ≈ 0", abs(peak_time_ccrps(perfect, target)["scalar"]) < 1e-9)
    # deterministic peak shifted +3 h → angular distance 3
    shifted = np.roll(diurnal, 3)
    shifted_da = hourly_dataarray(np.tile(shifted, (1, 3, 1)), lat, lon).isel(ensemble=0, drop=True)
    peak_offset = peak_time_ccrps(shifted_da, target)["scalar"]
    check("PTE-cCRPS +3h shift == 3", abs(peak_offset - 3.0) < 1e-6, f"got {peak_offset:.3f}")

    # ── Tail block ───────────────────────────────────────────────────────────
    print("Tail ratio / conditional rank hist / probabilistic detection")
    obs_tmax = xr.DataArray(rng.normal(300, 3, (400, 1, 1, 1)), dims=["forecast", "day_idx", "lat", "lon"],
                            coords={"forecast": np.arange(400), "day_idx": [0], "lat": [54.0], "lon": [10.0]})
    model_tmax = xr.DataArray(300 + 2 * (rng.normal(300, 3, (5, 400, 1, 1, 1)) - 300),
                              dims=["ensemble", "forecast", "day_idx", "lat", "lon"],
                              coords={"ensemble": np.arange(5), "forecast": np.arange(400),
                                      "day_idx": [0], "lat": [54.0], "lon": [10.0]})
    ratio = tail_ratio(model_tmax, obs_tmax)["ratio"]
    check("tail ratio ≈ 2 (double spread)", abs(ratio - 2.0) < 0.35, f"got {ratio:.3f}")

    members = xr.DataArray(rng.normal(300, 2, (8, 200, 1, 1, 1)),
                           dims=["ensemble", "forecast", "day_idx", "lat", "lon"],
                           coords={"ensemble": np.arange(8), "forecast": np.arange(200),
                                   "day_idx": [0], "lat": [54.0], "lon": [10.0]})
    obs_extreme = xr.full_like(members.isel(ensemble=0, drop=True), 320.0)  # above all members
    conditional = conditional_rank_histogram(members, obs_extreme, xr.DataArray(310.0))
    check("conditional rank hist: obs-above-all → top bin",
          conditional[-1] > 0.99, f"top bin {conditional[-1]:.3f}")

    positive_cells = xr.DataArray(np.concatenate([np.full((8, 100), 320.0), np.full((8, 100), 290.0)], axis=1)[..., None, None, None],
                                  dims=["ensemble", "forecast", "day_idx", "lat", "lon"],
                                  coords={"ensemble": np.arange(8), "forecast": np.arange(200),
                                          "day_idx": [0], "lat": [54.0], "lon": [10.0]})
    obs_binary = xr.DataArray(np.concatenate([np.full(100, 320.0), np.full(100, 290.0)])[..., None, None, None],
                              dims=["forecast", "day_idx", "lat", "lon"],
                              coords={"forecast": np.arange(200), "day_idx": [0], "lat": [54.0], "lon": [10.0]})
    detection = probabilistic_detection(positive_cells, obs_binary, xr.DataArray(310.0))
    check("Brier perfect == 0", abs(detection["brier"]) < 1e-9, f"BS {detection['brier']:.4f}")
    check("BSS perfect == 1", abs(detection["brier_skill_score"] - 1.0) < 1e-9)
    check("AUC perfect == 1", abs(detection["auc"] - 1.0) < 1e-9)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("Over-smoothing / equivalence")
    rough = hourly_dataarray(rng.normal(285, 4, (1, 3, 24)), lat, lon).isel(ensemble=0, drop=True)
    smooth = hourly_dataarray(np.tile(np.linspace(283, 289, 24), (1, 3, 1)), lat, lon).isel(ensemble=0, drop=True)
    smoothing = over_smoothing_diagnostic(smooth.expand_dims(ensemble=[0]), rough)
    check("over-smoothing: smooth curvature < rough",
          float(smoothing["pred_curve"].mean()) < float(smoothing["tgt_curve"].mean()))

    check("equivalence preserved", equivalence_test(-0.2, 0.05, 0.10)["verdict"] == "preserved")
    check("equivalence degraded", equivalence_test(0.02, 0.30, 0.10)["verdict"] == "degraded")
    check("equivalence inconclusive", equivalence_test(-0.2, 0.30, 0.10)["verdict"] == "inconclusive")

    # ── Windowed-p90 threshold logic (synthetic, no store) ───────────────────
    # Reproduce build_tmax_threshold's pool-then-percentile so the exact-window
    # computation is checked without touching LUMI-O.
    print("Windowed p90 threshold")
    window_days, days_per_year, years = 15, 365, 30
    latitude_cells, longitude_cells = 4, 5
    daily_max = rng.normal(295, 5, (years, days_per_year, latitude_cells, longitude_cells)).astype(np.float32)
    day_of_year = 196  # mid-July
    window_indices = [(day_of_year - 1 + offset) % days_per_year
                      for offset in range(-window_days, window_days + 1)]
    # Exactly build_tmax_threshold's pool-then-percentile: (year × 31, lat, lon) → p90 per cell.
    pool = daily_max[:, window_indices, :, :].reshape(-1, latitude_cells, longitude_cells)
    p90_map = np.percentile(pool, 90, axis=0)
    check("pool size == years × 31", pool.shape[0] == years * (2 * window_days + 1), f"{pool.shape[0]}")
    check("windowed p90 map shape == (lat, lon)", p90_map.shape == (latitude_cells, longitude_cells))
    check("windowed p90 within per-cell pool range",
          bool((pool.min(axis=0) <= p90_map).all() and (p90_map <= pool.max(axis=0)).all()))
    check("windowed p90 matches per-cell numpy percentile",
          float(np.abs(p90_map - np.percentile(pool, 90, axis=0)).max()) < 1e-4)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL PART I CHECKS PASSED")


if __name__ == "__main__":
    main()
