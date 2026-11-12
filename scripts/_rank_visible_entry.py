"""Per-rank entry point: bind Kit's renderer GPU to the rank's physical GPU.

``holosoma.train_agent_rank_visible`` narrows CUDA_VISIBLE_DEVICES to one GPU
per rank, so CUDA and PhysX see that GPU as ``cuda:0``. Kit's
``/renderer/activeGpu`` instead indexes every GPU on the machine, and Isaac
Lab sets it to 0 as well. On ranks other than the first physical GPU this
selects a device that CUDA cannot see: Kit creates no GPU foundation device,
and PhysX tensor views then fail ("Failed to get DOF velocities from backend").
Selecting the physical index keeps the renderer and PhysX on the same GPU.
"""

from __future__ import annotations

import os
import runpy


def _physical_gpu() -> str | None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.lower() == "all":
        return str(rank)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    return tokens[rank] if rank < len(tokens) else None


def main() -> None:
    if os.environ.get("HOLOSOMA_RANK_VISIBLE_DEVICES", "1").strip().lower() not in {"0", "false", "no", "off"}:
        gpu = _physical_gpu()
        # Kit indexes GPUs by ordinal; GPU UUID tokens keep the upstream default.
        if gpu is not None and gpu.isdigit():
            existing = os.environ.get("ISAACSIM_KIT_ARGS", "").strip()
            os.environ["ISAACSIM_KIT_ARGS"] = " ".join(filter(None, [existing, f"--/renderer/activeGpu={gpu}"]))
    runpy.run_module("holosoma.train_agent_rank_visible", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
