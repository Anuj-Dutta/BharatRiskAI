"""Merge downloaded DEM tiles without performing work during import."""
from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DEM_DIR = PROJECT_ROOT / "data" / "dem" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dem"
OUTPUT_FILE = OUTPUT_DIR / "kolkata_dem.tif"


def merge_dem(
    raw_dem_dir: Path = RAW_DEM_DIR,
    output_file: Path = OUTPUT_FILE,
) -> Path:
    """Merge all DEM tiles and return the generated raster path."""
    input_files = list(raw_dem_dir.glob("**/*.hgt")) + list(raw_dem_dir.glob("**/*.tif"))
    if not input_files:
        raise FileNotFoundError(
            f"No .hgt or .tif files found in {raw_dem_dir}!\n"
            "Ensure you have extracted your NASA Earthdata zip files there."
        )

    print(f"Found {len(input_files)} tile(s) to merge:")
    for input_file in input_files:
        print(f"  - {input_file.name}")

    src_files = [rasterio.open(input_file) for input_file in input_files]
    try:
        mosaic, out_transform = merge(src_files)
        metadata = src_files[0].meta.copy()
        metadata.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "count": mosaic.shape[0],
                "transform": out_transform,
                "crs": src_files[0].crs or CRS.from_epsg(4326),
                "compress": "lzw",
            }
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **metadata) as destination:
            destination.write(mosaic)
    finally:
        for source in src_files:
            source.close()

    print(f"\nMerged DEM saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    merge_dem()
