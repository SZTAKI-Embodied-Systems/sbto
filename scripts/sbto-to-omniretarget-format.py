#!/usr/bin/env python3
"""Convert an SBTO trajectory into the OmniRetarget-compatible qpos format.

The script reads a SBTO-produced ``best_trajectory.npz`` (or a run directory
containing it) and writes a new ``.npz`` file with

    qpos: (T, Nq) in OmniRetarget layout
    fps:  scalar frame rate in Hz

This matches the loader used by the holosoma conversion script when
``--use_omniretarget_data`` is enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _find_best_trajectory(input_path: Path) -> Path:
    if input_path.is_dir():
        matches = sorted(input_path.glob("**/best_trajectory.npz"))
        if not matches:
            raise FileNotFoundError(
                f"No best_trajectory.npz found under {input_path}. "
                "Provide the file directly or pass a run directory."
            )
        return matches[0]
    if input_path.is_file() and input_path.suffix == ".npz":
        return input_path
    raise FileNotFoundError(f"Unsupported input: {input_path}")


def _infer_fps(data: dict[str, Any], fallback: float) -> float:
    time = data.get("time")
    if time is not None:
        t = np.asarray(time, dtype=np.float64)
        if t.ndim == 1 and t.size > 1:
            dt = np.diff(t)
            positive = dt[dt > 0]
            if positive.size:
                return float(1.0 / np.median(positive))
    return float(fallback)


def _require_2d(name: str, arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected `{name}` to be 2D, got shape {arr.shape}")
    return arr


def _extract_qpos_from_sbto(data: dict[str, Any]) -> tuple[np.ndarray, float]:
    root_pos = data.get("root_pos")
    root_rot = data.get("root_rot")
    dof_pos = data.get("dof_pos")
    object_pos = data.get("object_pos")
    object_rot = data.get("object_rot")

    if root_pos is None or root_rot is None:
        if "x" not in data:
            raise KeyError(
                "SBTO file contains neither split-state keys nor a full `x` array; "
                "expected `root_pos`, `root_rot`, `dof_pos`, etc."
            )
        x = _require_2d("x", data["x"])
        # Best trajectory from the default SBTO setup stores split-state arrays,
        # but if only `x` is available we can still recover the qpos part.
        # The qpos layout is [root_pos(3), root_rot(4), dof_pos(N), object_pos(3), object_rot(4)].
        if object_pos is None and object_rot is None:
            object_is_present = False
        else:
            object_is_present = True

        if object_is_present:
            if object_pos is None or object_rot is None:
                raise KeyError("Found one object key but not both: object_pos and object_rot")
            root_pos = x[:, :3]
            root_rot = x[:, 3:7]
            dof_pos = x[:, 7:-7]
            object_pos = x[:, -7:-4]
            object_rot = x[:, -4:]
        else:
            root_pos = x[:, :3]
            root_rot = x[:, 3:7]
            dof_pos = x[:, 7:]
            object_pos = None
            object_rot = None

    root_pos = _require_2d("root_pos", root_pos)
    root_rot = _require_2d("root_rot", root_rot)
    dof_pos = _require_2d("dof_pos", dof_pos)

    if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(
            "Inconsistent frame counts across root_pos/root_rot/dof_pos: "
            f"{root_pos.shape[0]}, {root_rot.shape[0]}, {dof_pos.shape[0]}"
        )

    if object_pos is not None or object_rot is not None:
        object_pos = _require_2d("object_pos", object_pos)
        object_rot = _require_2d("object_rot", object_rot)
        if object_pos.shape[0] != root_pos.shape[0] or object_rot.shape[0] != root_pos.shape[0]:
            raise ValueError("Inconsistent frame count for object data")

    qpos_frames = []
    for i in range(root_pos.shape[0]):
        # SBTO state stores [pos, quat] for each free joint, while OmniRetarget expects [quat, pos].
        root_q = root_rot[i]
        root_p = root_pos[i]
        robot_q = dof_pos[i]
        if object_pos is not None and object_rot is not None:
            obj_q = object_rot[i]
            obj_p = object_pos[i]
            frame = np.concatenate((root_q, root_p, robot_q, obj_q, obj_p))
        else:
            frame = np.concatenate((root_q, root_p, robot_q))
        qpos_frames.append(frame)

    qpos = np.stack(qpos_frames, axis=0)
    fps = _infer_fps(data, fallback=50.0)
    return qpos, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="SBTO run directory or best_trajectory.npz file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output .npz file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the converted file. Defaults to the input file's "
            "directory; the output name appends "
            "'_to_omniretarget_format' to the input filename."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Fallback FPS used when the SBTO file does not contain a `time` array",
    )
    args = parser.parse_args()

    src = _find_best_trajectory(args.input)
    if args.output is not None and args.output_dir is not None:
        parser.error("--output and --output-dir cannot be used together")

    data = dict(np.load(src, allow_pickle=False))
    qpos, fps = _extract_qpos_from_sbto(data)
    fqps = float(args.fps if args.fps is not None else fps)
    if "time" in data:
        fqps = _infer_fps(data, fallback=fqps)

    if args.output is not None:
        output = args.output
    else:
        output_dir = args.output_dir if args.output_dir is not None else src.parent
        output = output_dir / f"{src.stem}_to_omniretarget_format.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=qpos, fps=np.array(fqps, dtype=np.float64))

    print(f"Converted SBTO trajectory from {src}")
    print(f"Saved OmniRetarget-compatible qpos to {output}")
    print(f"Frames={qpos.shape[0]}, qpos_shape={qpos.shape}, fps={fqps}")


if __name__ == "__main__":
    main()
