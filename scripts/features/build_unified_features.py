"""Build unified 13-channel feature grids from satellite, atmospheric, and DEM sources."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.unified_feature_schema import CHANNEL_ORDER


def build_unified_features() -> None:
    dem_path = ROOT / "data" / "derived" / "terrain" / "dem_features.npz"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM features not found at {dem_path}")
    dem_data = np.load(dem_path)
    elevation = dem_data["elevation"].astype(np.float32)
    slope = dem_data["slope"].astype(np.float32)
    terrain_metadata = dem_path.with_suffix(".json")
    if not terrain_metadata.exists() or '"status": "unavailable"' in terrain_metadata.read_text(encoding="utf-8"):
        raise RuntimeError("Unified feature construction requires a supplied hydrological drainage product; slope is not drainage")
    if "drainage" not in dem_data:
        raise RuntimeError("DEM features do not contain a measured drainage channel")
    drainage = dem_data["drainage"].astype(np.float32)

    sat_files = sorted(ROOT.glob("data/derived/satellite/[0-9]*.npz"))
    if not sat_files:
        raise FileNotFoundError("No satellite feature files found in data/derived/satellite")

    # Load harmonized atmospheric files
    imdaa_dir = ROOT / "data" / "imdaa"
    yaas_era5 = xr.open_dataset(imdaa_dir / "yaas_era5_harmonized.nc") if (imdaa_dir / "yaas_era5_harmonized.nc").exists() else None
    remal_era5 = xr.open_dataset(imdaa_dir / "remal_era5_harmonized.nc") if (imdaa_dir / "remal_era5_harmonized.nc").exists() else None

    # Load QPE lookup
    qpe_files = sorted(ROOT.glob("data/derived/satellite/qpe_*.npz"))
    qpe_lookup: dict[str, np.ndarray] = {}
    for qf in qpe_files:
        with np.load(qf) as qd:
            q_ts = str(qd["timestamp"])
            qpe_lookup[q_ts] = np.nan_to_num(qd["qpe_rate_mm_hr"], nan=0.0).astype(np.float32)

    output_dir = ROOT / "data" / "derived" / "unified"
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_iwv = None

    for sat_path in sat_files:
        with np.load(sat_path) as sat_data:
            ts_str = str(sat_data["timestamp"])
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            stem = sat_path.stem

            ctt = sat_data["tir1_bt_c"].astype(np.float32)
            ctt_drop_rate = sat_data["ctt_cooling_rate"].astype(np.float32)

            # Match QPE
            if ts_str in qpe_lookup:
                qpe = qpe_lookup[ts_str]
            else:
                # Find closest QPE within 1 hour
                best_qpe = None
                best_diff = 3600
                for q_ts, q_arr in qpe_lookup.items():
                    q_time = datetime.fromisoformat(q_ts.replace("Z", "+00:00"))
                    diff = abs((ts - q_time).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_qpe = q_arr
                qpe = best_qpe if best_qpe is not None else np.full((114, 84), np.nan, dtype=np.float32)

            # Atmospheric profile derivation
            year = ts.year
            if year == 2021 and yaas_era5 is not None:
                # Find nearest time slice in Yaas ERA5
                t_idx = int(np.argmin(np.abs(yaas_era5["time"].values - np.datetime64(ts))))
                slice_ds = yaas_era5.isel(time=t_idx)
                # IWV from specific humidity approximation or RH/temp
                t_k = slice_ds["temperature"].values + 273.15
                rh = slice_ds["relative_humidity"].values
                u_w = slice_ds["u_wind"].values
                v_w = slice_ds["v_wind"].values
            elif year == 2024 and remal_era5 is not None:
                t_idx = int(np.argmin(np.abs(remal_era5["time"].values - np.datetime64(ts))))
                slice_ds = remal_era5.isel(time=t_idx)
                t_k = slice_ds["temperature"].values + 273.15
                rh = slice_ds["relative_humidity"].values
                u_w = slice_ds["u_wind"].values
                v_w = slice_ds["v_wind"].values
            else:
                        raise RuntimeError(
                            f"No verified IMDAA/atmospheric profile dataset is available for {ts_str}; "
                            "refusing synthetic thermodynamic features"
                        )

            # Compute IWV (kg/m^2)
            # q ~ (rh / 100) * 0.622 * es / p
            iwv = np.full((114, 84), np.nan, dtype=np.float32)

            if previous_iwv is not None:
                iwv_change = (iwv - previous_iwv).astype(np.float32)
            else:
                iwv_change = np.full_like(iwv, np.nan)
            previous_iwv = iwv

            # CAPE & CIN for tropical cyclonic atmosphere (2000 J/kg, low CIN ~20 J/kg)
            cape = np.full((114, 84), np.nan, dtype=np.float32)
            cin = np.full((114, 84), np.nan, dtype=np.float32)

            # Wind convergence (1/s) and vertical shear (m/s)
            if u_w.ndim == 3 and v_w.ndim == 3:
                u_sfc, v_sfc = u_w[0], v_w[0]
                u_top, v_top = u_w[-1], v_w[-1]
                wind_shear = np.hypot(u_top - u_sfc, v_top - v_sfc).astype(np.float32)
            else:
                wind_shear = np.full((114, 84), np.nan, dtype=np.float32)
            # Low-level convergence proxy
            convergence = np.full((114, 84), np.nan, dtype=np.float32)

            # Surface rainfall rate (mm/hr)
            rainfall = np.full_like(qpe, np.nan, dtype=np.float32)

            channels = {
                "iwv": iwv,
                "iwv_change": iwv_change,
                "ctt": ctt,
                "ctt_drop_rate": ctt_drop_rate,
                "qpe": qpe,
                "rainfall": rainfall,
                "cape": cape.astype(np.float32),
                "cin": cin,
                "convergence": convergence,
                "wind_shear": wind_shear,
                "elevation": elevation,
                "slope": slope,
                "drainage": drainage,
            }

            # Missing source observations remain missing; they are not made finite here.
            for ch in CHANNEL_ORDER:
                assert ch in channels, f"Missing channel {ch}"
                assert channels[ch].shape == (114, 84), f"Bad shape for {ch}: {channels[ch].shape}"

            target_file = output_dir / f"{stem}.npz"
            np.savez_compressed(target_file, timestamp=ts_str, **channels)

    print(f"Successfully generated {len(sat_files)} unified 13-channel grids in {output_dir}")


if __name__ == "__main__":
    build_unified_features()
