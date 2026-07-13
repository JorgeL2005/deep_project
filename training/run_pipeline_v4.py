"""Short v4 round: recover counter_circuit / large_room regressions + custom-map eval.

- Starts from the v3 winner (bc_f3_iter6.pt); v3/v2 weight files untouched.
- 2 short DAgger iterations with x4 episode oversampling of the regressed
  layouts (counter_circuit, large_room) and x2 of the historically weak ones,
  stuck-state oversampling kept.
- Checkpoint selection by gameplay probe (return - 2*idle) with the v3 model
  as baseline guard: the exported model is never worse than v3.
- Exports policies/bc_weights_v4.npz.
- Evaluates BOTH v3 and v4 on ALL usable custom layouts (self + greedy, both
  roles, seeds 0-2) -> training/logs/custom_eval.json, plus the official
  matrix for v4 -> training/logs/final_eval_v4.json.

Usage:  vt\\Scripts\\python.exe -m training.run_pipeline_v4
"""

from __future__ import annotations

import importlib.util
import json
import time
import traceback

import numpy as np

from training.common import IN_CHANNELS, OFFICIAL_LAYOUTS, PAD_H, PAD_W, PROJECT_ROOT, build_layout_env
from training import rollouts as R
from training import train_bc as T
from training.run_pipeline import DATA_DIR, LOG_DIR, MODEL_DIR, log, short, usable_layouts
from training.run_pipeline_v3 import probe_v3

FOCUS_LAYOUTS = {"counter_circuit": 4, "large_room": 4}
WEAK_LAYOUTS = {"cramped_room": 2, "small_corridor": 2, "simple_tomato": 2, "forced_coordination": 2}


def gen_dagger_v4(layouts, model, device, out_path, rng):
    samples = []
    t0 = time.time()
    for li, layout in enumerate(layouts):
        name = short(layout)
        repeats = FOCUS_LAYOUTS.get(name, WEAK_LAYOUTS.get(name, 1))
        try:
            for rep in range(repeats):
                student = lambda: R.NetPolicy(model, device)  # noqa: E731
                s = 1000 + 100 * li + 10 * rep
                configs = [
                    (student(), student(), {0, 1}, {0: R.TeacherPolicy(seed=s), 1: R.TeacherPolicy(seed=s + 7)}),
                    (student(), R.TeacherPolicy(seed=s), {0}, {0: R.TeacherPolicy(seed=s + 13)}),
                    (student(), R.BuiltinPolicy("greedy_full_task", seed=s), {0}, {0: R.TeacherPolicy(seed=s + 3)}),
                    (R.BuiltinPolicy("greedy_full_task", seed=s), student(), {1}, {1: R.TeacherPolicy(seed=s + 4)}),
                    (student(), R.BuiltinPolicy("random_motion", seed=s), {0}, {0: R.TeacherPolicy(seed=s + 5)}),
                ]
                for pa, pb, record, shadows in configs:
                    smp, _ = R.rollout_collect(
                        layout, [pa, pb], record, rng,
                        exec_eps=0.02, shadow_teachers=shadows, stuck_boost=4,
                    )
                    samples.extend(smp)
            if (li + 1) % 10 == 0:
                log(f"  dagger-v4: {li + 1}/{len(layouts)} layouts, {len(samples)} samples, {time.time() - t0:.0f}s")
        except Exception:
            log(f"  dagger-v4 FAILED on {name}:\n{traceback.format_exc()}")
    R.save_samples(samples, out_path)
    log(f"  saved {len(samples)} samples -> {out_path}")


def _bc_policy_class(weights_path):
    spec = importlib.util.spec_from_file_location(
        "bc_agent_v4", PROJECT_ROOT / "policies" / "student_agent_bc.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class BCPolicy:
        def __init__(self, seed=0):
            self.agent = mod.StudentAgent({"weights": str(weights_path), "seed": seed})

        def start(self, env):
            self.agent.reset()

        def act(self, env, state, idx):
            obs = np.asarray(env.lossless_state_encoding_mdp(state)[idx])
            return int(self.agent.act(obs))

    return BCPolicy


def eval_matrix(layouts, weights_path, tag, partners=("self", "greedy_full_task")):
    BCPolicy = _bc_policy_class(weights_path)
    rng = np.random.default_rng(0)
    results = {}
    for layout in layouts:
        for partner in partners:
            rets = []
            for bc_idx in (0, 1):
                for seed in (0, 1, 2):
                    pols = [None, None]
                    pols[bc_idx] = BCPolicy(seed=seed)
                    if partner == "self":
                        pols[1 - bc_idx] = BCPolicy(seed=seed + 50)
                    else:
                        pols[1 - bc_idx] = R.BuiltinPolicy(partner, seed=seed)
                    try:
                        _, ret = R.rollout_collect(layout, pols, set(), rng)
                    except Exception:
                        ret = float("nan")
                    rets.append(ret)
                if partner == "self":
                    break
            mean = float(np.nanmean(rets))
            results[f"{short(layout)} | {partner}"] = {"mean": mean, "returns": rets}
            log(f"  [{tag}] {short(layout):45.45s} vs {partner:17s} mean={mean:7.1f}")
    return results


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== v4 short round start (device={device}) ===")
    rng = np.random.default_rng(7)
    layouts = usable_layouts()

    model = T.build_model()
    model.load_state_dict(torch.load(MODEL_DIR / "bc_f3_iter6.pt", map_location="cpu"))
    model = model.to(device)

    best = {"score": -1e9, "state": None, "tag": None}

    def consider(m, tag):
        ret, idle = probe_v3(m, device, rng)
        score = ret - 2.0 * idle
        log(f"  probe [{tag}] mean_return={ret:.1f} mean_max_idle={idle:.1f} score={score:.1f}")
        if score > best["score"]:
            best.update(score=score, tag=tag, state={k: v.cpu().clone() for k, v in m.state_dict().items()})

    consider(model, "v3-base")

    files = [DATA_DIR / "iter0_f3.npz", DATA_DIR / "iter5_f3.npz", DATA_DIR / "iter6_f3.npz"]
    for it in (7, 8):
        fi = DATA_DIR / f"iter{it}_f3.npz"
        if not fi.exists():
            log(f"v4 stage {it}: DAgger rollouts (focus counter_circuit/large_room)...")
            gen_dagger_v4(layouts, model, device, fi, rng)
        else:
            log(f"v4 stage {it}: iter{it}_f3.npz exists, reusing")
        files.append(fi)
        log(f"v4 stage {it}: fine-tuning on {len(files)} files...")
        groups = R.load_sample_files(files)
        model = T.train_model(groups, epochs=6, lr=2e-4, device=device, init_model=model, log=log)
        torch.save(model.state_dict(), MODEL_DIR / f"bc_f3_iter{it}.pt")
        consider(model, f"iter{it}")

    log(f"best v4 checkpoint: {best['tag']}")
    model = T.build_model()
    model.load_state_dict(best["state"])

    weights_path = PROJECT_ROOT / "policies" / "bc_weights_v4.npz"
    arrays = T.export_npz(model, weights_path)
    x = np.random.default_rng(1).random((IN_CHANNELS, PAD_H, PAD_W)).astype(np.float32)
    ref = model.cpu().eval()
    with torch.no_grad():
        want = ref(torch.from_numpy(x[None]))[0].numpy()
    err = float(np.abs(want - T.numpy_forward(arrays, x)).max())
    log(f"  numpy parity max abs err = {err:.2e}")
    assert err < 1e-3

    # Official matrix for v4.
    log("v4 official-layout evaluation...")
    off = eval_matrix(OFFICIAL_LAYOUTS, weights_path, "v4-official",
                      partners=("self", "greedy_full_task", "random_motion"))
    (LOG_DIR / "final_eval_v4.json").write_text(json.dumps(off, indent=1))

    # Custom-map evaluation: v3 vs v4 on every usable custom layout.
    customs = [l for l in layouts if str(l).endswith(".layout")]
    log(f"custom-map evaluation on {len(customs)} layouts (v3 then v4)...")
    v3_res = eval_matrix(customs, PROJECT_ROOT / "policies" / "bc_weights_v3.npz", "v3-custom")
    v4_res = eval_matrix(customs, weights_path, "v4-custom")
    (LOG_DIR / "custom_eval.json").write_text(json.dumps({"v3": v3_res, "v4": v4_res}, indent=1))

    m3 = float(np.nanmean([v["mean"] for v in v3_res.values()]))
    m4 = float(np.nanmean([v["mean"] for v in v4_res.values()]))
    log(f"custom-map means: v3={m3:.1f}  v4={m4:.1f}")
    log("=== v4 pipeline done ===")


if __name__ == "__main__":
    main()
