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

    atmospheric_dir = ROOT / "data" / "processed" / "atmospheric_features"

    # Load QPE lookup
    qpe_files = sorted(ROOT.glob("data/derived/satellite/qpe_*.npz"))
    qpe_lookup: dict[str, np.ndarray] = {}
    for qf in qpe_files:
        with np.load(qf) as qd:
            q_ts = str(qd["timestamp"])
            qpe_lookup[q_ts] = qd["qpe_rate_mm_hr"].astype(np.float32)

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

            atmospheric_path = atmospheric_dir / f"atmospheric_{ts.strftime('%Y%m%dT%H%M%SZ')}.nc"
            if not atmospheric_path.exists():
                raise RuntimeError(f"No derived IMDAA atmospheric product for {ts_str}: {atmospheric_path}")
            with xr.open_dataset(atmospheric_path) as atmospheric:
                iwv = np.asarray(atmospheric["iwv"].values, dtype=np.float32)
                cape = np.asarray(atmospheric["cape"].values, dtype=np.float32)
                cin = np.asarray(atmospheric["cin"].values, dtype=np.float32)
                convergence = np.asarray(atmospheric["horizontal_convergence"].values, dtype=np.float32)
                wind_shear = np.asarray(atmospheric["wind_shear_0_6km"].values, dtype=np.float32)

            if previous_iwv is not None:
                iwv_change = (iwv - previous_iwv).astype(np.float32)
            else:
                iwv_change = np.full_like(iwv, np.nan)
            previous_iwv = iwv

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
