"""Continued training (v3): fix idle/freeze behavior without touching v2.

Starts from the best v2 checkpoint and runs 3 more DAgger iterations with:
  - stuck-state oversampling (student frozen >=4 steps while the teacher says
    act -> sample duplicated x4),
  - 3x episode oversampling of the weak layouts,
  - warm-started fine-tuning (lower LR) on the aggregate dataset,
  - checkpoint selection by real gameplay INCLUDING an idle-streak metric,
  - baseline guard: the current v2 model competes in the selection, so the
    exported model is never worse than what we already have.

Output: policies/bc_weights_v3.npz (bc_weights.npz is NOT modified).

Usage:  vt\\Scripts\\python.exe -m training.run_pipeline_v3
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
)
from training import rollouts as R
from training import train_bc as T
from training.run_pipeline import DATA_DIR, LOG_DIR, MODEL_DIR, log, short, usable_layouts

WEAK_LAYOUTS = [
    "cramped_room",
    "small_corridor",
    "simple_tomato",
    "forced_coordination",
    "coordination_ring",
]

PROBE_LAYOUTS = [
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "forced_coordination",
    "soup_coordination",
    "small_corridor",
    "simple_tomato",
]


def probe_v3(model, device, rng):
    """Play probe games; return (mean_return, mean_max_idle_streak)."""
    from training.common import build_layout_env
    from src.constants import action_index_to_overcooked_action

    rets, idles = [], []
    for layout in PROBE_LAYOUTS:
        for partner in ("self", "greedy_full_task"):
            for net_idx in (0, 1):
                pols = [None, None]
                pols[net_idx] = R.NetPolicy(model, device)
                if partner == "self":
                    pols[1 - net_idx] = R.NetPolicy(model, device)
                else:
                    pols[1 - net_idx] = R.BuiltinPolicy(partner, seed=net_idx)
                env = build_layout_env(layout, horizon=400)
                env.reset(regen_mdp=False)
                for i, pol in enumerate(pols):
                    pol.start(env)
                    if isinstance(pol, R.BuiltinPolicy):
                        pol.set_index(i, env)
                total, done = 0.0, False
                streak, max_streak, last_pos = 0, 0, None
                while not done:
                    state = env.state
                    acts = [p.act(env, state, i) for i, p in enumerate(pols)]
                    pos = state.players[net_idx].position
                    if pos == last_pos:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                    last_pos = pos
                    joint = tuple(action_index_to_overcooked_action(a) for a in acts)
                    _, r, done, _ = env.step(joint)
                    total += float(r)
                rets.append(total)
                idles.append(max_streak)
    return float(np.mean(rets)), float(np.mean(idles))


def gen_dagger_v3(layouts, model, device, out_path, rng):
    samples = []
    t0 = time.time()
    weak = set(WEAK_LAYOUTS)
    for li, layout in enumerate(layouts):
        repeats = 3 if layout in weak else 1
        try:
            for rep in range(repeats):
                student = lambda: R.NetPolicy(model, device)  # noqa: E731
                s = 100 * li + 10 * rep
                configs = [
                    (student(), student(), {0, 1}, {0: R.TeacherPolicy(seed=s), 1: R.TeacherPolicy(seed=s + 7)}),
                    (student(), R.TeacherPolicy(seed=s), {0}, {0: R.TeacherPolicy(seed=s + 13)}),
                    (student(), R.BuiltinPolicy("greedy_full_task", seed=s), {0}, {0: R.TeacherPolicy(seed=s + 3)}),
                    (R.BuiltinPolicy("greedy_full_task", seed=s), student(), {1}, {1: R.TeacherPolicy(seed=s + 4)}),
                    (student(), R.BuiltinPolicy("random_motion", seed=s), {0}, {0: R.TeacherPolicy(seed=s + 5)}),
                    (R.BuiltinPolicy("stay", seed=s), student(), {1}, {1: R.TeacherPolicy(seed=s + 6)}),
                ]
                for pa, pb, record, shadows in configs:
                    smp, _ = R.rollout_collect(
                        layout, [pa, pb], record, rng,
                        exec_eps=0.02, shadow_teachers=shadows, stuck_boost=4,
                    )
                    samples.extend(smp)
            if (li + 1) % 10 == 0:
                log(f"  dagger-v3: {li + 1}/{len(layouts)} layouts, {len(samples)} samples, {time.time() - t0:.0f}s")
        except Exception:
            log(f"  dagger-v3 FAILED on {short(layout)}:\n{traceback.format_exc()}")
    R.save_samples(samples, out_path)
    log(f"  saved {len(samples)} samples -> {out_path}")


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== v3 continued training start (device={device}) ===")
    rng = np.random.default_rng(42)
    layouts = usable_layouts()
    log(f"usable layouts: {len(layouts)}")

    # Start from the best v2 checkpoint.
    model = T.build_model()
    model.load_state_dict(torch.load(MODEL_DIR / "bc_f3_iter3.pt", map_location="cpu"))
    model = model.to(device)

    best = {"score": -1e9, "state": None, "tag": None, "ret": 0, "idle": 0}

    def consider(m, tag):
        ret, idle = probe_v3(m, device, rng)
        score = ret - 2.0 * idle  # returns dominate; idle streaks tie-break hard
        log(f"  probe [{tag}] mean_return={ret:.1f} mean_max_idle={idle:.1f} score={score:.1f}")
        if score > best["score"]:
            best.update(
                score=score, tag=tag, ret=ret, idle=idle,
                state={k: v.cpu().clone() for k, v in m.state_dict().items()},
            )

    consider(model, "v2-base")  # baseline guard: never export something worse

    base_files = [DATA_DIR / "iter0_f3.npz", DATA_DIR / "iter3_f3.npz"]
    new_files = []
    for it in (4, 5, 6):
        fi = DATA_DIR / f"iter{it}_f3.npz"
        if not fi.exists():
            log(f"v3 stage {it}: DAgger rollouts (stuck_boost=4, weak x3)...")
            gen_dagger_v3(layouts, model, device, fi, rng)
        else:
            log(f"v3 stage {it}: iter{it}_f3.npz exists, reusing")
        new_files.append(fi)
        files = base_files + new_files[-3:]
        log(f"v3 stage {it}: fine-tuning on {len(files)} files...")
        groups = R.load_sample_files(files)
        model = T.train_model(groups, epochs=8, lr=3e-4, device=device, init_model=model, log=log)
        torch.save(model.state_dict(), MODEL_DIR / f"bc_f3_iter{it}.pt")
        consider(model, f"iter{it}")

    log(f"best v3 checkpoint: {best['tag']} (return {best['ret']:.1f}, idle {best['idle']:.1f})")
    model = T.build_model()
    model.load_state_dict(best["state"])

    weights_path = PROJECT_ROOT / "policies" / "bc_weights_v3.npz"
    arrays = T.export_npz(model, weights_path)
    x = np.random.default_rng(1).random((IN_CHANNELS, PAD_H, PAD_W)).astype(np.float32)
    ref = model.cpu().eval()
    with torch.no_grad():
        want = ref(torch.from_numpy(x[None]))[0].numpy()
    got = T.numpy_forward(arrays, x)
    err = float(np.abs(want - got).max())
    log(f"  numpy parity max abs err = {err:.2e}")
    assert err < 1e-3

    # Final matrix of the v3 agent (bc_weights.npz untouched).
    from training.run_pipeline import final_eval

    # final_eval writes final_eval.json; preserve the v2 record first.
    v2_json = LOG_DIR / "final_eval.json"
    v2_copy = LOG_DIR / "final_eval_v2.json"
    if v2_json.exists() and not v2_copy.exists():
        v2_copy.write_text(v2_json.read_text())

    log("v3 final evaluation (exported v3 weights)...")
    results = final_eval(OFFICIAL_LAYOUTS, weights_path)
    (LOG_DIR / "final_eval_v3.json").write_text(json.dumps(results, indent=1))
    log("=== v3 pipeline done ===")


if __name__ == "__main__":
    main()
