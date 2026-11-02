"""The Phase-0 falsification matrix.

The question this module answers is *not* "does belief steering work". It is:

    Before building a belief-steering decoder, find out whether the signal
    exists, where it lives, and whether *any* intervention on it changes the
    model's answers. A negative answer here is a result, not a failure.

The matrix crosses four things:

    mechanism   none | prompt_oracle | act | margin | logit | margin_shuffled
    layer       the depth at which the belief subspace is read and intervened on
    alpha       the intervention strength
    target      the real belief labels vs. a shuffled-label control

The ``margin_shuffled`` arm is the control that matters most: it re-runs the
margin intervention with a subspace fitted on **randomly permuted** labels. If
the shuffled arm performs like the real arm, the real arm is measuring generic
"confidence" steering and the belief claim is void.

Cost control: only the ``act`` arms need the model to be re-run (the
intervention changes the forward pass). The ``margin`` and ``logit`` arms act
on cached quantities and are therefore free, which is what makes a
6-layer x 3-alpha sweep affordable on a laptop.

Resumability: the expensive base pass is cached to an ``.npz`` keyed by
(model, items, seed, layers), and each ``act`` arm writes its own JSON as it
completes. A killed run resumes instead of starting over.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import ToMItem, generate_items, render_candidate, render_prompt
from .evaluate import bootstrap_ci, error_breakdown, paired_test, tom_accuracy
from .intervene import ActivationSteerer
from .model import decoder_layers, load_model, score_item
from .subspace import BeliefSubspace, fit_subspace, normalize_margins

MECHANISMS: tuple[str, ...] = (
    "none",
    "prompt_oracle",
    "act",
    "margin",
    "logit",
    "margin_shuffled",
)


@dataclass
class MatrixConfig:
    """Configuration for the Phase-0 matrix."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_items: int = 200
    seed: int = 0
    layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24)
    act_alphas: tuple[float, ...] = (1.0,)
    score_alphas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    probe_kind: str = "lda"
    out_dir: str = "results"
    device: str | None = None
    layer_scale: float = 1.0
    skip_existing: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


@dataclass
class Arm:
    """One cell of the matrix."""

    mechanism: str
    layer: int | None
    alpha: float

    @property
    def label(self) -> str:
        if self.layer is None:
            return f"{self.mechanism}"
        return f"{self.mechanism}-L{self.layer}-a{self.alpha:g}"

    def to_json(self) -> dict:
        return {"mechanism": self.mechanism, "layer": self.layer, "alpha": self.alpha, "label": self.label}


def build_arms(config: MatrixConfig) -> list[Arm]:
    """Enumerate every arm of the matrix in a deterministic order."""
    arms: list[Arm] = [Arm("none", None, 0.0), Arm("prompt_oracle", None, 0.0)]
    for layer in config.layers:
        for alpha in config.act_alphas:
            arms.append(Arm("act", layer, alpha))
    for mech in ("margin", "logit", "margin_shuffled"):
        for layer in config.layers:
            for alpha in config.score_alphas:
                arms.append(Arm(mech, layer, alpha))
    return arms


def expected_arm_count(config: MatrixConfig) -> int:
    """Number of arms :func:`build_arms` produces, for the integrity check."""
    n = 2 + len(config.layers) * len(config.act_alphas)
    n += 3 * len(config.layers) * len(config.score_alphas)
    return n


def oracle_prompt(item: ToMItem) -> str:
    """Prompt that states the protagonist's belief outright.

    This is an **oracle**: it uses ground truth the deployed system would not
    have. It is an upper bound on what prompt-level belief injection can buy,
    and the results document must say so every time this arm is quoted.
    """
    belief = item.options[item.correct_index]
    header = f"Note: {item.protagonist} believes the {item.obj} is in the {belief}."
    return f"{header}\n\n{render_prompt(item)}"


def _label_matrix(items: list[ToMItem]) -> np.ndarray:
    """Labels for every (item, candidate) pair: 1 = belief-consistent."""
    y = np.zeros(len(items) * 3, dtype=int)
    for i, item in enumerate(items):
        y[i * 3 + item.correct_index] = 1
    return y


def _gram_matrix(model, chunk: int = 32768) -> np.ndarray:
    """``G = W^T W`` where ``W`` is the language-model head weight.

    This lets the logit-bias mechanism be evaluated from the cached final
    hidden state instead of caching a full vocabulary logit vector per
    candidate: ``logits @ u = h_norm @ (G @ n)``.

    The matmul is chunked over the vocabulary and kept in float32. Casting the
    full ``(vocab, d)`` head to float64 would allocate ~1.1 GB and OOM a 3 GB
    machine -- a bug the pilot caught.
    """
    head = model.get_output_embeddings()
    W = head.weight.detach().float().cpu().numpy()
    G = np.zeros((W.shape[1], W.shape[1]), dtype=np.float64)
    for s in range(0, W.shape[0], chunk):
        part = W[s : s + chunk]
        G += part.T.astype(np.float64) @ part.astype(np.float64)
    return G


def _final_norm_weight(model) -> np.ndarray | None:
    """Weight of the final RMSNorm, if the architecture has one."""
    for path in (("model", "norm"), ("transformer", "ln_f"), ("model", "final_layernorm")):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "weight"):
            return obj.weight.detach().float().cpu().numpy().astype(np.float64)
    return None


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _cache_key(config: MatrixConfig) -> str:
    return f"{config.model_id}|n{config.n_items}|s{config.seed}|L{'-'.join(map(str, config.layers))}"


def run_matrix(config: MatrixConfig, verbose: bool = True) -> dict:
    """Run the Phase-0 matrix and write one JSON file per arm.

    Returns
    -------
    dict
        Summary with the configuration, the fitted subspaces, the arm results
        and the timing.
    """
    t_start = time.time()
    out_dir = Path(config.out_dir)
    arm_dir = out_dir / "ablation"
    arm_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "base_cache.npz"

    items = generate_items(config.n_items, config.seed)
    layers = list(config.layers)

    if verbose:
        print(f"[matrix] model={config.model_id} items={config.n_items} layers={layers}", flush=True)

    model, tokenizer = load_model(config.model_id, config.device)
    device = str(next(model.parameters()).device)
    n_model_layers = len(decoder_layers(model))
    if verbose:
        print(f"[matrix] loaded on {device} ({n_model_layers} decoder layers)", flush=True)

    # ---------------------------------------------------------------- base pass
    base_logprobs = oracle_logprobs = states = final_states = None
    exact = True
    key = _cache_key(config)

    if config.skip_existing and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    base_logprobs = z["base_logprobs"]
                    oracle_logprobs = z["oracle_logprobs"]
                    states = z["states"]
                    final_states = z["final_states"]
                    exact = bool(z["exact"])
                    if verbose:
                        print(f"[matrix] base pass loaded from cache ({cache_path})", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"[matrix] cache unreadable ({exc}); recomputing", flush=True)

    if base_logprobs is None:
        t0 = time.time()
        bl, ol, st, fs = [], [], [], []
        for i, item in enumerate(items):
            prompt = render_prompt(item)
            cands = [render_candidate(item, j) for j in range(3)]
            bs = score_item(model, tokenizer, prompt, cands, layers, device)
            os_ = score_item(model, tokenizer, oracle_prompt(item), cands, layers, device)
            bl.append([c.logprob for c in bs.candidates])
            ol.append([c.logprob for c in os_.candidates])
            st.append([[c.states[layer] for layer in layers] for c in bs.candidates])
            fs.append([[c.final_state for c in bs.candidates]][0])
            exact = exact and bs.exact_tokenisation
            if verbose and (i + 1) % 25 == 0:
                print(f"[matrix] base pass {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        base_logprobs = np.array(bl, dtype=np.float64)
        oracle_logprobs = np.array(ol, dtype=np.float64)
        states = np.array(st, dtype=np.float32)          # (n, 3, L, d)
        final_states = np.array(fs, dtype=np.float32)    # (n, 3, d)
        np.savez_compressed(cache_path, key=np.array(key), base_logprobs=base_logprobs,
                            oracle_logprobs=oracle_logprobs, states=states,
                            final_states=final_states, exact=np.array(exact))
        if verbose:
            print(f"[matrix] base pass done in {time.time()-t0:.0f}s; exact tokenisation={exact}", flush=True)
    base_pass_s = time.time() - t_start

    # --------------------------------------------------------------- subspaces
    y = _label_matrix(items)
    rng = np.random.default_rng(config.seed + 1)
    subspaces: dict[int, BeliefSubspace] = {}
    subspaces_shuffled: dict[int, BeliefSubspace] = {}
    for li, layer in enumerate(layers):
        X = states[:, :, li, :].reshape(-1, states.shape[-1]).astype(np.float64)
        subspaces[layer] = fit_subspace(X, y, layer=layer, kind=config.probe_kind, seed=config.seed)
        subspaces_shuffled[layer] = fit_subspace(
            X, rng.permutation(y), layer=layer, kind=config.probe_kind, seed=config.seed
        )

    # ------------------------------------------------------------- score arms
    base_pred = base_logprobs.argmax(axis=1)
    correct = np.array([it.correct_index for it in items])
    results: dict[str, dict] = {}

    def record(arm: Arm, pred: np.ndarray, extra: dict | None = None) -> None:
        breakdown = error_breakdown(pred, items)
        acc = tom_accuracy(pred, items)
        per_item = (pred == correct).astype(float)
        lo, hi = bootstrap_ci(per_item, seed=config.seed)
        stat, pval = paired_test(per_item, (base_pred == correct).astype(float))
        payload = {
            "arm": arm.to_json(),
            "n_items": len(items),
            "accuracy": acc,
            "accuracy_ci95": [lo, hi],
            "error_breakdown": breakdown,
            "world_rate": breakdown["world"] / len(items),
            "other_rate": breakdown["other"] / len(items),
            "vs_none_wilcoxon": {"statistic": stat, "p_value": pval},
            # The per-item predictions are stored so that any statistic can be
            # recomputed later, exactly, without re-running the model.
            "predictions": [int(v) for v in pred],
        }
        if extra:
            payload.update(extra)
        results[arm.label] = payload
        (arm_dir / f"{arm.label}.json").write_text(json.dumps(payload, indent=2))
        if verbose:
            print(
                f"[matrix] {arm.label:28s} acc={acc:.3f} world={payload['world_rate']:.3f} "
                f"other={payload['other_rate']:.3f} p={pval:.3g}",
                flush=True,
            )

    record(Arm("none", None, 0.0), base_pred, {"note": "base model, no intervention"})
    record(Arm("prompt_oracle", None, 0.0), oracle_logprobs.argmax(axis=1),
           {"note": "ORACLE belief stated in the prompt; upper bound"})

    # activation steering arms: require re-running the model
    for layer in layers:
        sub = subspaces[layer]
        direction = _unit(sub.w)
        li = layers.index(layer)
        scale = config.layer_scale * float(
            np.mean(np.linalg.norm(states[:, :, li, :].reshape(-1, states.shape[-1]).astype(np.float64), axis=1))
        )
        for alpha in config.act_alphas:
            arm = Arm("act", layer, alpha)
            f = arm_dir / f"{arm.label}.json"
            if config.skip_existing and f.exists():
                results[arm.label] = json.loads(f.read_text())
                if verbose:
                    print(f"[matrix] {arm.label:28s} (cached)", flush=True)
                continue
            steerer = ActivationSteerer(direction, layer=layer, alpha=alpha, scale=scale)
            preds = []
            t0 = time.time()
            for i, item in enumerate(items):
                sc = score_item(
                    model, tokenizer, render_prompt(item),
                    [render_candidate(item, j) for j in range(3)],
                    layers, device, steerer=steerer,
                )
                preds.append(int(np.argmax([c.logprob for c in sc.candidates])))
                if verbose and (i + 1) % 50 == 0:
                    print(f"[matrix]   act {arm.label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
            record(arm, np.array(preds),
                   {"probe": {"train_acc": sub.train_acc, "heldout_acc": sub.heldout_acc}, "scale": scale})

    # margin and logit arms: computed from cached quantities
    G = _gram_matrix(model)
    gamma = _final_norm_weight(model)
    for li, layer in enumerate(layers):
        sub = subspaces[layer]
        sub_shuf = subspaces_shuffled[layer]
        X = states[:, :, li, :].reshape(-1, states.shape[-1]).astype(np.float64)
        marg = normalize_margins(sub.margin(X)).reshape(len(items), 3)
        marg_shuf = normalize_margins(sub_shuf.margin(X)).reshape(len(items), 3)

        if gamma is not None:
            rms = np.linalg.norm(sub.w) / np.sqrt(sub.w.shape[0])
            n_vec = (sub.w / (rms + 1e-12)) * gamma
            g_vec = G @ n_vec
            bias = (final_states.reshape(-1, final_states.shape[-1]).astype(np.float64) @ g_vec).reshape(len(items), 3)
        else:
            bias = np.zeros((len(items), 3))

        for alpha in config.score_alphas:
            record(Arm("margin", layer, alpha), (base_logprobs + alpha * marg).argmax(axis=1),
                   {"probe": {"train_acc": sub.train_acc, "heldout_acc": sub.heldout_acc}})
            record(Arm("margin_shuffled", layer, alpha), (base_logprobs + alpha * marg_shuf).argmax(axis=1),
                   {"probe": {"train_acc": sub_shuf.train_acc, "heldout_acc": sub_shuf.heldout_acc},
                    "note": "CONTROL: labels shuffled"})
            record(Arm("logit", layer, alpha), (base_logprobs + alpha * bias).argmax(axis=1),
                   {"probe": {"train_acc": sub.train_acc, "heldout_acc": sub.heldout_acc}})

    # ------------------------------------------------------------------ summary
    summary = {
        "config": config.to_json(),
        "device": device,
        "model_layers": n_model_layers,
        "exact_tokenisation": exact,
        "timing_sec": {"base_pass": round(base_pass_s, 1), "total": round(time.time() - t_start, 1)},
        "subspaces": {
            str(layer): {
                "kind": subspaces[layer].kind,
                "train_acc": subspaces[layer].train_acc,
                "heldout_acc": subspaces[layer].heldout_acc,
                "w_norm": float(np.linalg.norm(subspaces[layer].w)),
                "shuffled_heldout_acc": subspaces_shuffled[layer].heldout_acc,
            }
            for layer in layers
        },
        "arms": results,
    }
    (out_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[matrix] wrote {out_dir/'phase0_summary.json'} in {time.time()-t_start:.0f}s", flush=True)
    return summary
