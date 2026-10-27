#!/usr/bin/env python3
"""Regenerate the committed 10 s EuRoC smoke fixture from a full EuRoC download.

The fixture is the only dataset the closed-loop test suite can run against with no
$DATA_ROOT and no network, so it is committed (~631 KiB). This script is what makes
that legitimate: the fixture is reproducible rather than a blob of unknown origin.

    PYTHONPATH=. python Scripts/make_smoke_fixture.py

Rows are copied VERBATIM as text — never parsed and re-serialised. Round-tripping
EuRoC's 17-significant-digit floats through Python would perturb the last digits, and
the point of the fixture is to be the same bytes the loader would have read from the
real download. ``--check`` asserts exactly that against what is already committed.

Slicing rule (each number in the emitted PROVENANCE.txt is a consequence of it):

    IMU   the first ``duration_sec`` worth of rows at or after the first GT timestamp.
          EuRoC IMU leads GT by ~0.9 s, so starting at the first GT sample is what lets
          every frame be interpolated rather than extrapolated.
    GT    from the file start through the last kept IMU timestamp plus ``gt_margin_sec``.
          The margin is what keeps the *last* IMU frame interpolated too; without it
          EuRoC_Sequence clips the tail to the overlap and N comes out short.

At EuRoC's 200 Hz the defaults give exactly 2000 IMU rows and 2100 GT rows, which is
the ``smoke.frames == 2000`` that every window-size claim in the smoke dataset YAMLs is
arithmetic on.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

IMU_REL = os.path.join("mav0", "imu0", "data.csv")
GT_REL = os.path.join("mav0", "state_groundtruth_estimate0", "data.csv")
SENSOR_RELS = (
    os.path.join("mav0", "imu0", "sensor.yaml"),
    os.path.join("mav0", "state_groundtruth_estimate0", "sensor.yaml"),
)


def read_csv_lines(path: str) -> tuple[bytes, list[bytes]]:
    """Split a EuRoC CSV into its ``#``-comment header and its data lines, as bytes.

    Bytes, and splitting on ``\\n`` alone, because EuRoC ships CRLF: any ``\\r`` stays
    inside the line and is written back out untouched. Decoding to text and rejoining
    with ``\\n`` would silently rewrite every line ending, which is the difference
    between a verbatim copy and a reformatted one.
    """
    with open(path, "rb") as handle:
        lines = handle.read().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()                      # trailing terminator, not an empty row
    header = b""
    while lines and lines[0].startswith(b"#"):
        header += lines.pop(0) + b"\n"
    return header, lines


def timestamp_of(line: bytes) -> int:
    """First column of a EuRoC CSV row: an integer nanosecond timestamp."""
    return int(line.split(b",", 1)[0])


def slice_indices(
    t_imu: list[int], t_gt: list[int], duration_sec: float, gt_margin_sec: float
) -> tuple[int, int, int]:
    """``(imu_start, imu_stop, gt_stop)`` for the rule in the module docstring."""
    first_gt = t_gt[0]
    imu_start = next(i for i, t in enumerate(t_imu) if t >= first_gt)
    # Half-open in TIME: keep t - t_start < duration. At 200 Hz the row landing exactly
    # on the horizon is the 2001st, so an inclusive bound would give 2001 frames and
    # break the smoke YAMLs' window arithmetic (which assumes exactly 2000).
    horizon = t_imu[imu_start] + int(duration_sec * 1e9)
    imu_stop = next(
        (i for i in range(imu_start, len(t_imu)) if t_imu[i] >= horizon), len(t_imu)
    )
    # The GT bound IS inclusive: the margin exists to cover the last IMU frame, so a
    # GT sample landing exactly on it is the one doing the covering.
    gt_horizon = t_imu[imu_stop - 1] + int(gt_margin_sec * 1e9)
    gt_stop = next((j for j, t in enumerate(t_gt) if t > gt_horizon), len(t_gt))
    return imu_start, imu_stop, gt_stop


def provenance(
    seq_name: str,
    duration_sec: float,
    gt_margin_sec: float,
    imu_start: int,
    imu_stop: int,
    gt_stop: int,
    t_imu: list[int],
    t_gt: list[int],
) -> str:
    return (
        "EuRoC smoke-test fixture — rows copied verbatim, no reformatting.\n"
        "source_dataset  : EuRoC MAV (ETH Zurich), https://doi.org/10.3929/ethz-b-000690084\n"
        "license         : In Copyright - Non-Commercial Use Permitted; not covered by SSIO's MIT License\n"
        'citation        : Burri et al., "The EuRoC micro aerial vehicle datasets", '
        "IJRR 35(10), 2016\n"
        f"source_sequence : {seq_name}\n"
        "generator       : Scripts/make_smoke_fixture.py\n"
        "start_sec       : 0.0   (after the first GT sample)\n"
        f"duration_sec    : {duration_sec}\n"
        f"gt_margin_sec   : {gt_margin_sec}\n"
        f"imu_rows        : {imu_stop - imu_start} "
        f"(source rows [{imu_start}, {imu_stop}))\n"
        f"imu_t_ns        : [{t_imu[imu_start]}, {t_imu[imu_stop - 1]}]\n"
        f"gt_rows         : {gt_stop} (source rows [0, {gt_stop}))\n"
        f"gt_t_ns         : [{t_gt[0]}, {t_gt[gt_stop - 1]}]\n"
        "Regenerate with: PYTHONPATH=. python Scripts/make_smoke_fixture.py\n"
    )


def main() -> int:
    default_out = os.path.join(_REPO_ROOT, "tests", "fixtures", "euroc_mh02_10s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--euroc-root",
        default=None,
        help="directory holding the full EuRoC sequences "
        "(default: ${host_env:DATA_ROOT}/EuRoC)",
    )
    parser.add_argument("--seq-name", default="MH_02_easy")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--gt-margin-sec", type=float, default=0.5)
    parser.add_argument("--out", default=default_out, help="fixture root to write")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; assert the committed fixture is byte-identical to a "
        "fresh slice and exit non-zero if not",
    )
    args = parser.parse_args()

    euroc_root = args.euroc_root
    if euroc_root is None:
        # Imported here, not at module scope: it registers OmegaConf resolvers and
        # pulls in torch, which --help should not have to pay for.
        from Interface import host_env

        euroc_root = os.path.join(host_env("DATA_ROOT"), "EuRoC")

    src = os.path.join(euroc_root, args.seq_name)
    imu_src, gt_src = os.path.join(src, IMU_REL), os.path.join(src, GT_REL)
    for path in (imu_src, gt_src):
        if not os.path.isfile(path):
            parser.error(f"source CSV not found: {path}\n(is --euroc-root correct?)")

    imu_header, imu_lines = read_csv_lines(imu_src)
    gt_header, gt_lines = read_csv_lines(gt_src)
    t_imu = [timestamp_of(line) for line in imu_lines]
    t_gt = [timestamp_of(line) for line in gt_lines]

    imu_start, imu_stop, gt_stop = slice_indices(
        t_imu, t_gt, args.duration_sec, args.gt_margin_sec
    )
    imu_out = imu_header + b"\n".join(imu_lines[imu_start:imu_stop]) + b"\n"
    gt_out = gt_header + b"\n".join(gt_lines[:gt_stop]) + b"\n"
    prov_out = provenance(
        args.seq_name, args.duration_sec, args.gt_margin_sec,
        imu_start, imu_stop, gt_stop, t_imu, t_gt,
    ).encode()

    dst = os.path.join(args.out, args.seq_name)
    targets = {
        os.path.join(dst, IMU_REL): imu_out,
        os.path.join(dst, GT_REL): gt_out,
        os.path.join(dst, "PROVENANCE.txt"): prov_out,
    }

    if args.check:
        failures = []
        for path, expected in targets.items():
            if not os.path.isfile(path):
                failures.append(f"missing: {path}")
                continue
            with open(path, "rb") as handle:
                if handle.read() != expected:
                    failures.append(f"differs from a fresh slice: {path}")
        for rel in SENSOR_RELS:
            if not os.path.isfile(os.path.join(dst, rel)):
                failures.append(f"missing: {os.path.join(dst, rel)}")
        for line in failures:
            print(f"[make_smoke_fixture] {line}")
        print(
            "[make_smoke_fixture] "
            + ("FAIL" if failures else "OK — committed fixture is reproducible")
        )
        return 1 if failures else 0

    for path, content in targets.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
    # sensor.yaml is metadata the loader never reads, but EuRoC_Sequence's own
    # FileNotFoundError messages point at a mav0 tree, so keep the shape faithful.
    for rel in SENSOR_RELS:
        shutil.copyfile(os.path.join(src, rel), os.path.join(dst, rel))

    print(
        f"[make_smoke_fixture] wrote {imu_stop - imu_start} IMU rows and {gt_stop} GT "
        f"rows to {dst}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
