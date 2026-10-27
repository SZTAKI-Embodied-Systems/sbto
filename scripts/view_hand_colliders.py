"""View the G1 rubber hands and collision geoms; press R or Ctrl+R to reload."""

import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "sbto/models/unitree_g1/scene_mjx_29dof_no_hands.xml"
CONTACT_XML = ROOT / "sbto/models/unitree_g1/contact_pairs/full_hand.xml"


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    contacts = mujoco.MjSpec.from_file(str(CONTACT_XML))

    for pair in contacts.pairs:
        scene.add_pair(
            name=pair.name,
            geomname1=pair.geomname1,
            geomname2=pair.geomname2,
            condim=pair.condim,
            solref=pair.solref,
            solreffriction=pair.solreffriction,
            solimp=pair.solimp,
            margin=pair.margin,
            gap=pair.gap,
            friction=pair.friction,
            info=pair.info,
        )

    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def main() -> None:
    model, data = load_model()

    while True:
        reload_requested = threading.Event()

        def key_callback(keycode: int) -> None:
            if keycode == ord("R"):
                reload_requested.set()

        replacement = None
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.opt.geomgroup[3] = 1

            while viewer.is_running():
                if reload_requested.is_set():
                    try:
                        replacement = load_model()
                    except (OSError, RuntimeError, ValueError) as error:
                        print(f"Reload failed; keeping current model: {error}")
                        reload_requested.clear()
                    else:
                        print("Reloaded model from XML.")
                        break

                viewer.sync()
                time.sleep(0.02)

        if replacement is None:
            break
        model, data = replacement


if __name__ == "__main__":
    main()
