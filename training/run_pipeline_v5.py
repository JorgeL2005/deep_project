"""v5: specialize for the competition maps without losing generality.

Competition maps: asymmetric_advantages, coordination_ring, counter_circuit.
User-reported issue: the v3 agent is very SLOW on asymmetric_advantages.

Strategy (v3 weights untouched -> exports policies/bc_weights_v5.npz):
  - Continued DAgger from the v3 winner with x6 episode oversampling of the
    3 competition maps (all 57 layouts stay in the mix for generality) and
    stuck-state oversampling x4 (attacks slowness/hesitation directly).
  - Extra partner variety on the competition maps (self, teacher, greedy,
    random_motion, stay), both roles.
  - Checkpoint selection by composite gameplay score:
        score = comp_ret - 2*comp_idle + 0.4 * (gen_ret - 2*gen_idle)
    where comp_* is measured on the 3 competition maps and gen_* on the
    broader v3 probe set. The v3 model competes as baseline, so the export
    is never worse on this criterion.

Usage:  vt\\Scripts\\python.exe -m training.run_pipeline_v5
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np

from training.common import IN_CHANNELS, OFFICIAL_LAYOUTS, PAD_H, PAD_W, PROJECT_ROOT, build_layout_env
from training import rollouts as R
from training import train_bc as T
from training.run_pipeline import DATA_DIR, LOG_DIR, MODEL_DIR, log, short, usable_layouts
from training.run_pipeline_v3 import probe_v3
from training.run_pipeline_v4 import eval_matrix

from src.constants import action_index_to_overcooked_action

COMP_LAYOUTS = ["asymmetric_advantages", "coordination_ring", "counter_circuit"]
COMP_REPEATS = 6


def probe_comp(model, device):
    """Gameplay probe restricted to the competition maps.

    Partners: self, greedy, random_motion; both roles; 2 seeds via policy rng.
    Returns (mean_return, mean_max_idle).
    """
    rets, idles = [], []
    for layout in COMP_LAYOUTS:
        for partner in ("self", "greedy_full_task", "random_motion"):
            for net_idx in (0, 1):
                pols = [None, None]
                pols[net_idx] = R.NetPolicy(model, device)
                if partner == "self":
                    pols[1 - net_idx] = R.NetPolicy(model, device)
                else:
                    pols[1 - net_idx] = R.BuiltinPolicy(partner, seed=net_idx + 3)
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


def gen_dagger_v5(layouts, model, device, out_path, rng):
    samples = []
    t0 = time.time()
    comp = set(COMP_LAYOUTS)
    for li, layout in enumerate(layouts):
        name = short(layout)
        repeats = COMP_REPEATS if name in comp else 1
        try:
            for rep in range(repeats):
                student = lambda: R.NetPolicy(model, device)  # noqa: E731
                s = 5000 + 100 * li + 10 * rep
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
                log(f"  dagger-v5: {li + 1}/{len(layouts)} layouts, {len(samples)} samples, {time.time() - t0:.0f}s")
        except Exception:
            log(f"  dagger-v5 FAILED on {name}:\n{traceback.format_exc()}")
    R.save_samples(samples, out_path)
    log(f"  saved {len(samples)} samples -> {out_path}")


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== v5 competition-map specialization start (device={device}) ===")
    rng = np.random.default_rng(99)
    layouts = usable_layouts()

    model = T.build_model()
    model.load_state_dict(torch.load(MODEL_DIR / "bc_f3_iter6.pt", map_location="cpu"))  # v3 winner
    model = model.to(device)

    best = {"score": -1e9, "state": None, "tag": None}

    def consider(m, tag):
        comp_ret, comp_idle = probe_comp(m, device)
        gen_ret, gen_idle = probe_v3(m, device, rng)
        score = (comp_ret - 2.0 * comp_idle) + 0.4 * (gen_ret - 2.0 * gen_idle)
        log(
            f"  probe [{tag}] comp_ret={comp_ret:.1f} comp_idle={comp_idle:.1f} "
            f"gen_ret={gen_ret:.1f} gen_idle={gen_idle:.1f} score={score:.1f}"
        )
        if score > best["score"]:
            best.update(score=score, tag=tag,
                        state={k: v.cpu().clone() for k, v in m.state_dict().items()})

    consider(model, "v3-base")

    files = [DATA_DIR / "iter0_f3.npz", DATA_DIR / "iter5_f3.npz", DATA_DIR / "iter6_f3.npz"]
    for it in (9, 10, 11):
        fi = DATA_DIR / f"iter{it}_f3.npz"
        if not fi.exists():
            log(f"v5 stage {it}: DAgger rollouts (competition maps x{COMP_REPEATS})...")
            gen_dagger_v5(layouts, model, device, fi, rng)
        else:
            log(f"v5 stage {it}: iter{it}_f3.npz exists, reusing")
        files.append(fi)
        use = files[:2] + files[-3:]  # iter0 + iter5 + latest three
        log(f"v5 stage {it}: fine-tuning on {len(use)} files...")
        groups = R.load_sample_files(use)
        model = T.train_model(groups, epochs=8, lr=3e-4, device=device, init_model=model, log=log)
        torch.save(model.state_dict(), MODEL_DIR / f"bc_f3_iter{it}.pt")
        consider(model, f"iter{it}")

    log(f"best v5 checkpoint: {best['tag']}")
    model = T.build_model()
    model.load_state_dict(best["state"])

    weights_path = PROJECT_ROOT / "policies" / "bc_weights_v5.npz"
    arrays = T.export_npz(model, weights_path)
    x = np.random.default_rng(1).random((IN_CHANNELS, PAD_H, PAD_W)).astype(np.float32)
    ref = model.cpu().eval()
    with __import__("torch").no_grad():
        want = ref(__import__("torch").from_numpy(x[None]))[0].numpy()
    err = float(np.abs(want - T.numpy_forward(arrays, x)).max())
    log(f"  numpy parity max abs err = {err:.2e}")
    assert err < 1e-3

    # Detailed evaluation: competition maps (v5 vs v3), plus official matrix
    # for the generality check.
    log("v5 evaluation on competition maps (v3 first, then v5)...")
    v3w = PROJECT_ROOT / "policies" / "bc_weights_v3.npz"
    comp_v3 = eval_matrix(COMP_LAYOUTS, v3w, "v3-comp",
                          partners=("self", "greedy_full_task", "random_motion", "stay"))
    comp_v5 = eval_matrix(COMP_LAYOUTS, weights_path, "v5-comp",
                          partners=("self", "greedy_full_task", "random_motion", "stay"))
    log("v5 official-layout matrix (generality)...")
    off_v5 = eval_matrix(OFFICIAL_LAYOUTS, weights_path, "v5-official",
                         partners=("self", "greedy_full_task", "random_motion"))
    (LOG_DIR / "final_eval_v5.json").write_text(
        json.dumps({"comp_v3": comp_v3, "comp_v5": comp_v5, "official_v5": off_v5}, indent=1)
    )
    m3 = float(np.nanmean([v["mean"] for v in comp_v3.values()]))
    m5 = float(np.nanmean([v["mean"] for v in comp_v5.values()]))
    mo = float(np.nanmean([v["mean"] for v in off_v5.values()]))
    log(f"competition-map means: v3={m3:.1f}  v5={m5:.1f} | v5 official mean={mo:.1f}")
    log("=== v5 pipeline done ===")


if __name__ == "__main__":
    main()
