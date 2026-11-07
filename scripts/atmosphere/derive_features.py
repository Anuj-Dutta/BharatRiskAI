"""Derive timestamped atmospheric features from supplied IMDAA products.

The command is metadata-first and never creates values for unavailable
pressure-level or surface observations.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ml.atmosphere.features import (
    bulk_wind_shear_0_6km,
    surface_based_cape_cin,
    horizontal_convergence,
    integrated_water_vapour,
    specific_humidity_from_relative_humidity,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "temperature": ("TMP-prl", "temperature", "ta"),
    "relative_humidity": ("RH-prl", "relative_humidity", "rh"),
    "u_wind": ("UGRD-prl", "u_wind", "ua"),
    "v_wind": ("VGRD-prl", "v_wind", "va"),
    "height": ("HGT-prl", "geopotential_height", "zg"),
}


def _timestamp(dataset: xr.Dataset) -> datetime:
    for name in ("valid_time", "time", "forecast_time"):
        if name in dataset.coords and dataset[name].size:
            value = dataset[name].values.reshape(-1)[0]
            if isinstance(value, np.datetime64):
                value = value.astype("datetime64[us]").astype(datetime)
            if not isinstance(value, datetime):
                value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raise ValueError("Dataset has no valid timestamp coordinate")


def _find(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in dataset.variables), None)


def _units(dataset: xr.Dataset, name: str) -> str:
    units = str(dataset[name].attrs.get("units", "")).strip()
    if not units:
        raise ValueError(f"Variable {name!r} has no units")
    return units


def derive_profile_dataset(pressure: xr.Dataset, single: xr.Dataset | None = None) -> xr.Dataset:
    names = {key: _find(pressure, candidates) for key, candidates in REQUIRED.items()}
    missing = [key for key, name in names.items() if name is None]
    if missing:
        raise ValueError(f"Pressure-level dataset is missing variables: {missing}")
    level_name = next((name for name in ("isobaricInhPa", "pressure", "level", "plev") if name in pressure.coords), None)
    lat_name = next((name for name in ("latitude", "lat") if name in pressure.coords), None)
    lon_name = next((name for name in ("longitude", "lon") if name in pressure.coords), None)
    if not level_name or not lat_name or not lon_name:
        raise ValueError("Pressure-level dataset must expose pressure, latitude and longitude coordinates")
    level = np.asarray(pressure[level_name].values, dtype=float)
    level_pa = level * 100.0 if np.nanmedian(level) < 2_000 else level
    temperature = np.asarray(pressure[names["temperature"]].squeeze().values, dtype=float)
    rh = np.asarray(pressure[names["relative_humidity"]].squeeze().values, dtype=float)
    u = np.asarray(pressure[names["u_wind"]].squeeze().values, dtype=float)
    v = np.asarray(pressure[names["v_wind"]].squeeze().values, dtype=float)
    height = np.asarray(pressure[names["height"]].squeeze().values, dtype=float)
    _units(pressure, names["temperature"]); _units(pressure, names["relative_humidity"])
    _units(pressure, names["u_wind"]); _units(pressure, names["v_wind"]); _units(pressure, names["height"])
    q = specific_humidity_from_relative_humidity(temperature, rh, level_pa[:, None, None])
    iwv = integrated_water_vapour(q, level_pa)
    cape, cin = surface_based_cape_cin(temperature, rh, level_pa)
    shear, shear_u, shear_v = bulk_wind_shear_0_6km(u, v, height)
    convergence = horizontal_convergence(
        u[0], v[0], np.asarray(pressure[lat_name].values), np.asarray(pressure[lon_name].values)
    )
    result = xr.Dataset(
        {
            "iwv": (("latitude", "longitude"), iwv, {"units": "kg m-2", "long_name": "integrated water vapour"}),
            "cape": (("latitude", "longitude"), cape, {"units": "J kg-1", "long_name": "surface-based convective available potential energy"}),
            "cin": (("latitude", "longitude"), cin, {"units": "J kg-1", "long_name": "surface-based convective inhibition"}),
            "wind_shear_0_6km": (("latitude", "longitude"), shear, {"units": "m s-1", "long_name": "0-6 km bulk wind shear"}),
            "wind_shear_u_0_6km": (("latitude", "longitude"), shear_u, {"units": "m s-1"}),
            "wind_shear_v_0_6km": (("latitude", "longitude"), shear_v, {"units": "m s-1"}),
            "horizontal_convergence": (("latitude", "longitude"), convergence, {"units": "s-1", "long_name": "horizontal wind convergence"}),
        },
        coords={"latitude": pressure[lat_name].values, "longitude": pressure[lon_name].values},
        attrs={
            "feature_schema_version": "atmospheric-v1",
            "source_valid_time": _timestamp(pressure).isoformat(),
            "iwv_method": "trapezoidal integral of RH-derived specific humidity dp/g",
            "cape_cin_method": "MetPy surface-based parcel profile; CIN retains the library sign convention",
            "wind_shear_method": "height-interpolated 0-6 km bulk vector shear",
            "convergence_method": "spherical latitude/longitude divergence",
        },
    )
    if single is not None:
        result.attrs["single_level_source_valid_time"] = _timestamp(single).isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-dir", type=Path, default=ROOT / "data" / "imdaa" / "pressure_level")
    parser.add_argument("--single-dir", type=Path, default=ROOT / "data" / "imdaa" / "single_level")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "atmospheric_features")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    files = sorted(args.pressure_dir.glob("*.grib*")) + sorted(args.pressure_dir.glob("*.nc*"))
    if not files:
        raise SystemExit(f"No pressure-level IMDAA files found in {args.pressure_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "atmospheric_feature_manifest.json"
    manifest: dict[str, Any] = {"schema_version": "atmospheric-v1", "records": []}
    for path in files:
        try:
            with xr.open_dataset(path, engine="cfgrib" if path.suffix.startswith(".grib") else None) as dataset:
                timestamp = _timestamp(dataset)
                output = args.output_dir / f"atmospheric_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.nc"
                if output.exists() and not args.force:
                    status = "SKIPPED_EXISTING"
                else:
                    result = derive_profile_dataset(dataset)
                    result.to_netcdf(output)
                    status = "ACCEPTED"
            manifest["records"].append({"source": str(path), "timestamp": timestamp.isoformat(), "output": str(output), "status": status})
        except (OSError, ValueError, RuntimeError) as exc:
            manifest["records"].append({"source": str(path), "status": "BLOCKED", "reason": str(exc)})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
