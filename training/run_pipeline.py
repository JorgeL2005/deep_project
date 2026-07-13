"""Full BC + DAgger distillation pipeline.

Usage:
    python -m training.run_pipeline

Stages:
  0. Discover usable layouts (official + custom that build and fit 24x16).
  1. Iter 0: teacher rollouts (self-play + vs baselines) -> BC dataset.
  2. Train CNN.
  3. DAgger iters: roll out the student, relabel with the teacher, retrain.
  4. Export weights to policies/bc_weights.npz + numpy parity check.
  5. Final evaluation matrix of the trained agent (torch-free path).

All progress goes to training/logs/pipeline.log.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from training.common import (
    IN_CHANNELS,
    OFFICIAL_LAYOUTS,
    PAD_H,
    PAD_W,
    PROJECT_ROOT,
    build_layout_env,
    list_custom_layouts,
    pad_obs,
)
from training import rollouts as R
from training import train_bc as T

LOG_DIR = PROJECT_ROOT / "training" / "logs"
DATA_DIR = PROJECT_ROOT / "training" / "data"
MODEL_DIR = PROJECT_ROOT / "training" / "models"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_file = open(LOG_DIR / "pipeline.log", "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")
    _log_file.flush()


def usable_layouts():
    layouts = []
    for layout in OFFICIAL_LAYOUTS + list_custom_layouts():
        try:
            env = build_layout_env(layout, horizon=40)
            env.reset(regen_mdp=False)
            obs = np.asarray(env.lossless_state_encoding_mdp(env.state)[0])
            if obs.shape[0] <= PAD_W and obs.shape[1] <= PAD_H and obs.shape[2] == 26:
                layouts.append(layout)
            else:
                log(f"skip (too big {obs.shape}): {layout}")
        except Exception as exc:
            log(f"skip (build failed): {Path(str(layout)).name}: {exc!r}")
    return layouts


def short(layout):
    return Path(str(layout)).stem


def gen_teacher_data(layouts, out_path, rng):
    samples = []
    t0 = time.time()
    for li, layout in enumerate(layouts):
        try:
            # Self-play (record both sides), varied seeds.
            for s in range(3):
                pols = [R.TeacherPolicy(seed=100 * li + s), R.TeacherPolicy(seed=100 * li + s + 50)]
                smp, ret = R.rollout_collect(layout, pols, {0, 1}, rng, exec_eps=0.04)
                samples.extend(smp)
            # Vs baselines, both roles (record teacher side only).
            for partner in ("greedy_full_task", "random_motion", "stay"):
                for teacher_idx in (0, 1):
                    pols = [None, None]
                    pols[teacher_idx] = R.TeacherPolicy(seed=li)
                    pols[1 - teacher_idx] = R.BuiltinPolicy(partner, seed=li)
                    smp, ret = R.rollout_collect(layout, pols, {teacher_idx}, rng, exec_eps=0.04)
                    samples.extend(smp)
            if (li + 1) % 10 == 0:
                log(f"  teacher data: {li + 1}/{len(layouts)} layouts, {len(samples)} samples, {time.time() - t0:.0f}s")
        except Exception:
            log(f"  teacher data FAILED on {short(layout)}:\n{traceback.format_exc()}")
    R.save_samples(samples, out_path)
    log(f"  saved {len(samples)} samples -> {out_path}")


def gen_dagger_data(layouts, model, device, out_path, rng):
    samples = []
    t0 = time.time()
    for li, layout in enumerate(layouts):
        try:
            student = lambda: R.NetPolicy(model, device)  # noqa: E731
            configs = [
                # (policyA, policyB, record, shadows)
                (student(), student(), {0, 1}, {0: R.TeacherPolicy(seed=li), 1: R.TeacherPolicy(seed=li + 7)}),
                (student(), R.TeacherPolicy(seed=li), {0}, {0: R.TeacherPolicy(seed=li + 13)}),
                (student(), R.BuiltinPolicy("greedy_full_task", seed=li), {0}, {0: R.TeacherPolicy(seed=li + 3)}),
                (R.BuiltinPolicy("greedy_full_task", seed=li), student(), {1}, {1: R.TeacherPolicy(seed=li + 4)}),
                (student(), R.BuiltinPolicy("random_motion", seed=li), {0}, {0: R.TeacherPolicy(seed=li + 5)}),
                (R.BuiltinPolicy("stay", seed=li), student(), {1}, {1: R.TeacherPolicy(seed=li + 6)}),
            ]
            for pa, pb, record, shadows in configs:
                smp, ret = R.rollout_collect(
                    layout, [pa, pb], record, rng, exec_eps=0.02, shadow_teachers=shadows
                )
                samples.extend(smp)
            if (li + 1) % 10 == 0:
                log(f"  dagger data: {li + 1}/{len(layouts)} layouts, {len(samples)} samples, {time.time() - t0:.0f}s")
        except Exception:
            log(f"  dagger data FAILED on {short(layout)}:\n{traceback.format_exc()}")
    R.save_samples(samples, out_path)
    log(f"  saved {len(samples)} samples -> {out_path}")


def final_eval(layouts, weights_path):
    """Evaluate the exported numpy agent (the actual deliverable)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bc_agent", PROJECT_ROOT / "policies" / "student_agent_bc.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class BCPolicy:
        def __init__(self):
            self.agent = mod.StudentAgent({"weights": str(weights_path)})

        def start(self, env):
            self.agent.reset()

        def act(self, env, state, idx):
            obs = np.asarray(env.lossless_state_encoding_mdp(state)[idx])
            return int(self.agent.act(obs))

    rng = np.random.default_rng(0)
    results = {}
    for layout in layouts:
        for partner in ("self", "greedy_full_task", "random_motion"):
            rets = []
            for bc_idx in (0, 1):
                for seed in (0, 1, 2):
                    pols = [None, None]
                    pols[bc_idx] = BCPolicy()
                    if partner == "self":
                        pols[1 - bc_idx] = BCPolicy()
                    else:
                        pols[1 - bc_idx] = R.BuiltinPolicy(partner, seed=seed)
                    _, ret = R.rollout_collect(layout, pols, set(), rng)
                    rets.append(ret)
                if partner == "self":
                    break  # symmetric
            results[f"{short(layout)} | {partner}"] = {
                "mean": float(np.mean(rets)),
                "returns": rets,
            }
            log(f"  eval {short(layout):30s} vs {partner:18s} mean={np.mean(rets):7.1f}")
    (LOG_DIR / "final_eval.json").write_text(json.dumps(results, indent=1))
    return results


PROBE_LAYOUTS = [
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "forced_coordination",
    "soup_coordination",
]


def gameplay_probe(model, device, rng):
    """Quick real-play score of a checkpoint (mean sparse return)."""
    rets = []
    for layout in PROBE_LAYOUTS:
        for partner in ("self", "greedy_full_task"):
            for net_idx in (0, 1):
                pols = [None, None]
                pols[net_idx] = R.NetPolicy(model, device)
                if partner == "self":
                    pols[1 - net_idx] = R.NetPolicy(model, device)
                else:
                    pols[1 - net_idx] = R.BuiltinPolicy(partner, seed=net_idx)
                _, ret = R.rollout_collect(layout, pols, set(), rng, exec_eps=0.01)
                rets.append(ret)
    return float(np.mean(rets))


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== BC/DAgger pipeline v2 start (device={device}, frames=3) ===")
    rng = np.random.default_rng(0)

    layouts = usable_layouts()
    log(f"usable layouts: {len(layouts)}")

    data_files = []
    best = {"score": -1.0, "state": None, "tag": None}

    def consider(model, tag):
        score = gameplay_probe(model, device, rng)
        log(f"  gameplay probe [{tag}] mean return = {score:.1f}")
        if score > best["score"]:
            best.update(score=score, tag=tag, state={k: v.cpu().clone() for k, v in model.state_dict().items()})

    # ---- Iter 0: teacher data + BC ----
    f0 = DATA_DIR / "iter0_f3.npz"
    if not f0.exists():
        log("stage 1: generating teacher dataset (frame-stacked)...")
        gen_teacher_data(layouts, f0, rng)
    else:
        log("stage 1: iter0_f3.npz exists, reusing")
    data_files.append(f0)

    log("stage 2: BC training (iter 0)...")
    groups = R.load_sample_files(data_files)
    model = T.train_model(groups, epochs=20, device=device, log=log)
    torch.save(model.state_dict(), MODEL_DIR / "bc_f3_iter0.pt")
    consider(model, "iter0")

    # ---- DAgger iterations ----
    for it in (1, 2, 3):
        fi = DATA_DIR / f"iter{it}_f3.npz"
        if not fi.exists():
            log(f"stage 3.{it}: DAgger rollouts (iter {it})...")
            gen_dagger_data(layouts, model, device, fi, rng)
        else:
            log(f"stage 3.{it}: iter{it}_f3.npz exists, reusing")
        data_files.append(fi)
        log(f"stage 3.{it}: retraining on aggregate ({len(data_files)} files)...")
        groups = R.load_sample_files(data_files)
        model = T.train_model(groups, epochs=12, device=device, log=log)
        torch.save(model.state_dict(), MODEL_DIR / f"bc_f3_iter{it}.pt")
        consider(model, f"iter{it}")

    log(f"best checkpoint by gameplay: {best['tag']} (score {best['score']:.1f})")
    model = T.build_model()
    model.load_state_dict(best["state"])
    model = model.to(device)

    # ---- Export + parity ----
    log("stage 4: exporting weights...")
    weights_path = PROJECT_ROOT / "policies" / "bc_weights.npz"
    arrays = T.export_npz(model, weights_path)
    x = np.random.default_rng(1).random((IN_CHANNELS, PAD_H, PAD_W)).astype(np.float32)
    ref = model.cpu().eval()
    with torch.no_grad():
        want = ref(torch.from_numpy(x[None]))[0].numpy()
    got = T.numpy_forward(arrays, x)
    err = float(np.abs(want - got).max())
    log(f"  numpy parity max abs err = {err:.2e}")
    assert err < 1e-3, "numpy forward mismatch"

    # ---- Final evaluation ----
    log("stage 5: final evaluation of the exported numpy agent...")
    final_eval(OFFICIAL_LAYOUTS, weights_path)
    log("=== pipeline done ===")


if __name__ == "__main__":
    main()
