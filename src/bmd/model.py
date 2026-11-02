"""Model loading, per-layer state extraction, and candidate scoring.

Everything here is CPU/MPS friendly and never requires CUDA. The scoring
routine uses a prefix KV cache so that the three candidate continuations of an
item share one prompt prefill -- this is what makes a 6-layer x 2-alpha
sweep affordable on a laptop.

Tokenisation boundary
---------------------
A candidate continuation is scored by concatenating the prompt token ids with
the candidate token ids. When the tokenisation of ``prompt + candidate`` starts
with exactly the tokenisation of ``prompt``, the candidate ids are taken from
that joint tokenisation (exact). Otherwise the candidate is tokenised on its
own and concatenated (an approximation, but applied identically to every
candidate of every item, so comparisons stay fair). The fallback rate is
reported by :func:`score_item` so it can be audited rather than assumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .intervene import ActivationSteerer


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Resolve a dtype name to a :class:`torch.dtype`.

    ``None`` keeps the historical default of ``float32``, which is what the
    recorded Phase-0 run used. The default is deliberately **not** changed to
    a smaller dtype: doing so would silently change the numbers of an already
    published run.

    ``bfloat16`` is the useful option on Apple Silicon for a 1.5B model
    (3.1 GB instead of 6.2 GB); ``float16`` is the fallback when the platform
    lacks bfloat16 support.
    """
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(mapping)} or None"
        )
    return mapping[name]


def load_model(
    model_id: str,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
):
    """Load a causal LM and its tokenizer.

    Parameters
    ----------
    model_id:
        HuggingFace model id, e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``.
    device:
        ``None`` selects ``mps`` when available, then ``cuda``, then ``cpu``.
    dtype:
        ``None`` (the default) loads in ``float32``, which is what the
        recorded Phase-0 run used. Pass ``"bfloat16"`` to halve the memory
        footprint of a larger model on Apple Silicon.

    Returns
    -------
    (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def decoder_layers(model) -> list:
    """Return the list of transformer decoder layer modules."""
    for path in (("model", "layers"), ("transformer", "h"), ("model", "decoder", "layers")):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return list(obj)
    raise AttributeError("could not locate the decoder layer list on this model")


@torch.no_grad()
def extract_states(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: str | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract the hidden state at the last token of each text.

    Parameters
    ----------
    texts:
        Texts to encode.
    layer:
        Index into ``output.hidden_states`` (``-1`` is the final layer).
    batch_size:
        Batch size.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(texts), d)``.
    """
    if device is None:
        device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        batch = {k: v.to(device) for k, v in batch.items()}
        hs = model(**batch, output_hidden_states=True).hidden_states[layer]
        lengths = batch["attention_mask"].sum(dim=1) - 1
        out.append(hs[torch.arange(hs.shape[0]), lengths].float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


@dataclass
class CandidateScore:
    """Scoring information for one candidate continuation."""

    logprob: float
    n_tokens: int
    states: dict[int, np.ndarray] = field(default_factory=dict)
    final_state: np.ndarray | None = None


@dataclass
class ItemScores:
    """Scores for all candidates of one item."""

    prompt: str
    candidates: list[CandidateScore]
    exact_tokenisation: bool = True


def _split_ids(tokenizer, prompt: str, candidate: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return ``(prompt_ids, candidate_ids, exact)``."""
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(prompt + candidate, return_tensors="pt").input_ids[0]
    if len(full_ids) > len(p_ids) and torch.equal(full_ids[: len(p_ids)], p_ids):
        return p_ids, full_ids[len(p_ids) :], True
    cand_ids = tokenizer(candidate, add_special_tokens=False).input_ids
    if not cand_ids:
        cand_ids = tokenizer(candidate).input_ids
    return p_ids, torch.tensor(cand_ids, dtype=p_ids.dtype), False


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _steer_ctx(model, steerer: ActivationSteerer | None):
    """Context manager installing the activation-steering hook, if any."""
    if steerer is None:
        return _NullContext()
    layers = decoder_layers(model)
    if not (-len(layers) <= steerer.layer < len(layers)):
        raise IndexError(f"layer {steerer.layer} out of range for {len(layers)} layers")
    module = layers[steerer.layer]
    scale = float(steerer.alpha * steerer.scale)
    vector = torch.tensor(steerer.direction, dtype=torch.float32)

    def fn(mod, args, output):  # noqa: ANN001
        v = (scale * vector).to(output[0].device if isinstance(output, tuple) else output.device)
        v = v.to((output[0] if isinstance(output, tuple) else output).dtype)
        if isinstance(output, tuple):
            return (output[0] + v,) + output[1:]
        return output + v

    class _Ctx:
        def __enter__(self):
            self.handle = module.register_forward_hook(fn)
            return self

        def __exit__(self, *exc):
            self.handle.remove()
            return False

    return _Ctx()


@torch.no_grad()
def score_item(
    model,
    tokenizer,
    prompt: str,
    candidates: list[str],
    layers: list[int],
    device: str | None = None,
    steerer: ActivationSteerer | None = None,
) -> ItemScores:
    """Score every candidate continuation of one prompt.

    Returns the total log-probability of each candidate under the model, the
    per-layer hidden state of the candidate's last token, and the final
    (post-norm) hidden state used by the logit-bias mechanism.
    """
    if device is None:
        device = next(model.parameters()).device

    encoded = [_split_ids(tokenizer, prompt, c) for c in candidates]
    exact = all(e[2] for e in encoded)
    p_ids = encoded[0][0]
    # Extract the candidate ids by index rather than by tuple unpacking: the
    # arity of `_split_ids`' return value is an implementation detail, and
    # unpacking it in the loop silently couples the two.
    cand_id_list = [e[1] for e in encoded]

    results: list[CandidateScore] = []
    with _steer_ctx(model, steerer):
        p_tensor = p_ids.unsqueeze(0).to(device)
        out = model(p_tensor, output_hidden_states=True, use_cache=True)
        past = out.past_key_values
        prompt_last_logits = out.logits[0, -1]

        for cand_ids in cand_id_list:
            c_tensor = cand_ids.unsqueeze(0).to(device)
            # The model appends the candidate tokens to the cache in place.
            # Each candidate must be scored against the SAME prompt context,
            # so it gets its own copy of the prompt cache. (Mutating and
            # cropping the shared cache is not portable: `DynamicCache.crop`
            # changed semantics between transformers releases.)
            c_out = model(c_tensor, past_key_values=copy.deepcopy(past), output_hidden_states=True, use_cache=True)
            cand_logits = c_out.logits[0]
            logprob = _logprob_of(prompt_last_logits, cand_ids, cand_logits)
            states = {ly: c_out.hidden_states[ly][0, -1].float().cpu().numpy() for ly in layers}
            final = c_out.hidden_states[-1][0, -1].float().cpu().numpy()
            results.append(
                CandidateScore(
                    logprob=logprob,
                    n_tokens=int(cand_ids.numel()),
                    states=states,
                    final_state=final,
                )
            )
    return ItemScores(prompt=prompt, candidates=results, exact_tokenisation=exact)


def _logprob_of(prompt_last_logits, cand_ids, cand_logits) -> float:
    """Total log-probability of the candidate tokens given the prompt."""
    logprobs = torch.log_softmax(prompt_last_logits.float(), dim=-1)
    total = float(logprobs[cand_ids[0]])
    if cand_ids.numel() > 1:
        step = torch.log_softmax(cand_logits[:-1].float(), dim=-1)
        total += float(step[torch.arange(len(cand_ids) - 1), cand_ids[1:]].sum())
    return total
