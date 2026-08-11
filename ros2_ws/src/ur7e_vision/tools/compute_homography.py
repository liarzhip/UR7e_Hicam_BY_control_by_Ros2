#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_points(csv_path):
    pixel = []
    base = []

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        for row in reader:
            if not row:
                continue
            pixel.append([float(row["u"]), float(row["v"])])
            base.append([float(row["x"]), float(row["y"])])

    if len(pixel) < 4:
        raise RuntimeError("At least 4 point correspondences are required.")

    return (
        np.asarray(pixel, dtype=np.float64),
        np.asarray(base, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    pixel, base = read_points(args.csv)

    H, mask = cv2.findHomography(pixel, base, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("cv2.findHomography failed.")

    pred = cv2.perspectiveTransform(pixel.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - base, axis=1)

    print("Homography pixel -> base XY:")
    print(H)
    print()
    print(f"Points: {len(pixel)}")
    print(f"Mean XY error: {err.mean()*1000:.3f} mm")
    print(f"Max  XY error: {err.max()*1000:.3f} mm")
    print(f"RANSAC inliers: {int(mask.sum()) if mask is not None else 'N/A'}")

    data = {
        "homography": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in H.reshape(-1)],
        },
        "calibration": {
            "point_count": int(len(pixel)),
            "mean_xy_error_m": float(err.mean()),
            "max_xy_error_m": float(err.max()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
