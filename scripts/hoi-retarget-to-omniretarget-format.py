#!/usr/bin/env python3
"""Convert HOI-Retarget Parquet motions to SBTO-compatible NPZ references.

The converter exports Unitree G1 rows. Each NPZ contains ``qpos`` and ``fps``;
the qpos layout matches SBTO's G1 robot-plus-object reference scene.

Requires pyarrow in addition to the project dependencies:
    python -m pip install pyarrow
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "datasets" / "hoi-retarget"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "hoi-retarget_converted"
DEFAULT_ROBOT_XML = (
    REPO_ROOT
    / "sbto"
    / "models"
    / "unitree_g1"
    / "g1_mjx_no_hands.xml"
)
ROBOT = "unitree_g1"
FPS = 30.0


def get_actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for joint_id in model.actuator_trnid[:, 0]:
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
        )
        if name is None:
            raise ValueError(f"Actuator references unnamed joint id {joint_id}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("The G1 model has duplicate actuated joint names")
    return names


def trajectory(row: dict[str, Any], key: str, width: int) -> np.ndarray:
    values = np.asarray(row[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(
            f"{row.get('clip_id', '<unknown>')}: expected {key} shape "
            f"(T, {width}), got {values.shape}"
        )
    return values


def make_qpos(
    row: dict[str, Any],
    model: mujoco.MjModel,
    target_joint_names: list[str],
) -> np.ndarray:
    clip_id = row["clip_id"]
    source_joint_names = list(row["dof_names"])
    dof_pos = trajectory(row, "dof_pos", len(source_joint_names))

    if len(source_joint_names) != len(set(source_joint_names)):
        raise ValueError(f"{clip_id}: duplicate names in dof_names")
    missing = sorted(set(target_joint_names) - set(source_joint_names))
    extra = sorted(set(source_joint_names) - set(target_joint_names))
    if missing or extra:
        raise ValueError(
            f"{clip_id}: joint names do not match the SBTO G1 model; "
            f"missing={missing}, extra={extra}"
        )

    root_pos = trajectory(row, "root_pos", 3)
    root_rot = trajectory(row, "root_rot", 4)
    object_pos = trajectory(row, "object_pos", 3)
    object_rot = trajectory(row, "object_rot", 4)

    frame_counts = {
        len(root_pos),
        len(root_rot),
        len(dof_pos),
        len(object_pos),
        len(object_rot),
    }
    if len(frame_counts) != 1:
        raise ValueError(f"{clip_id}: trajectory columns have different frame counts")

    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    if root_joint_id < 0:
        raise ValueError("Could not find floating_base_joint in the G1 model")

    root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
    qpos = np.zeros((len(root_pos), model.nq + 7), dtype=np.float64)

    # SBTO flips free joints from [quaternion, position] to [position, quaternion].
    qpos[:, root_qpos_adr : root_qpos_adr + 4] = root_rot
    qpos[:, root_qpos_adr + 4 : root_qpos_adr + 7] = root_pos

    source_indices = {name: index for index, name in enumerate(source_joint_names)}
    for joint_name in target_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"Could not find G1 joint {joint_name!r}")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos[:, qpos_adr] = dof_pos[:, source_indices[joint_name]]

    # The reference scene appends the object's free joint after the robot qpos.
    object_qpos_adr = model.nq
    qpos[:, object_qpos_adr : object_qpos_adr + 4] = object_rot
    qpos[:, object_qpos_adr + 4 : object_qpos_adr + 7] = object_pos

    if not np.isfinite(qpos).all():
        raise ValueError(f"{clip_id}: qpos contains non-finite values")
    return qpos


def output_filename(clip_id: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9._-]+", "__", clip_id).strip("._")
    if not filename:
        raise ValueError(f"Cannot create a filename from clip_id {clip_id!r}")
    return f"{filename}.npz"


def convert(input_dir: Path, output_dir: Path, robot_xml: Path) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet reading requires pyarrow. Install it with "
            "'python -m pip install pyarrow'."
        ) from exc

    parquet_files = sorted(input_dir.glob("data/*/*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {input_dir / 'data'}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"G1 MuJoCo model not found: {robot_xml}")

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    target_joint_names = get_actuated_joint_names(model)
    converted = 0
    total_rows = sum(
        parquet.ParquetFile(path).metadata.num_rows for path in parquet_files
    )

    with tqdm(
        total=total_rows,
        desc="Converting HOI-Retarget",
        unit="row",
        dynamic_ncols=True,
    ) as progress:
        for parquet_path in parquet_files:
            parquet_file = parquet.ParquetFile(parquet_path)
            progress.set_postfix_str(
                f"{parquet_path.parent.parent.name}/{parquet_path.parent.name}"
            )
            for batch in parquet_file.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    if row["robot"] == ROBOT:
                        qpos = make_qpos(row, model, target_joint_names)
                        output_path = (
                            output_dir
                            / row["dataset"]
                            / ROBOT
                            / output_filename(row["clip_id"])
                        )
                        if output_path.exists():
                            raise FileExistsError(
                                f"Refusing to overwrite existing output: "
                                f"{output_path}"
                            )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("xb") as output_file:
                            np.savez_compressed(
                                output_file,
                                qpos=qpos,
                                fps=np.float64(row.get("fps", FPS)),
                            )
                        converted += 1
                        progress.set_postfix(
                            shard=(
                                f"{parquet_path.parent.parent.name}/"
                                f"{parquet_path.parent.name}"
                            ),
                            converted=converted,
                        )
                    progress.update()

    if converted == 0:
        raise ValueError(f"No {ROBOT} rows found in {input_dir}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    args = parser.parse_args()

    converted = convert(args.input, args.output, args.robot_xml)
    print(f"Converted {converted} {ROBOT} motions to {args.output}")


if __name__ == "__main__":
    main()
