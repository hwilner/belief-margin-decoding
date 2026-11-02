# API Reference

Everything below runs on CPU. No CUDA is required or assumed.

---

## `bmd.data`

### `ToMItem`

```python
@dataclass(frozen=True)
class ToMItem:
    item_id: str
    story: str
    question: str
    protagonist: str
    obj: str
    options: tuple[str, str, str]   # candidate locations, in display order
    correct_index: int               # the protagonist's believed location
    world_index: int                 # the true location
    other_index: int                 # the third agent's believed location
```

`correct_index`, `world_index` and `other_index` are always pairwise distinct.

### Functions

```python
def generate_items(n: int, seed: int = 0) -> list[ToMItem]
```
Deterministic: two calls with the same `(n, seed)` return byte-identical items.
Raises `ValueError` if `n <= 0`.

```python
def render_prompt(item: ToMItem) -> str
def render_candidate(item: ToMItem, index: int) -> str   # " drawer."
def render_statement(item: ToMItem, index: int) -> str    # "Alice thinks the key is in the drawer."
def object_of(item: ToMItem) -> str
```

`render_candidate` raises `IndexError` for an index outside `0..2`.

---

## `bmd.subspace`

```python
@dataclass
class BeliefSubspace:
    w: np.ndarray            # (d,) direction
    b: float                 # bias
    layer: int
    train_acc: float         # internal 80/20 split
    heldout_acc: float
    kind: str                # "lda" | "logreg" | "mlp"
    n_train: int
    n_heldout: int
    meta: dict

    def margin(self, X: np.ndarray) -> np.ndarray   # (n, d) -> (n,)
    def predict(self, X: np.ndarray) -> np.ndarray   # 1 = belief-consistent
    def save(self, path) -> None                    # .npz
    @classmethod
    def load(cls, path) -> BeliefSubspace
```

```python
def fit_subspace(X, y, layer: int, reg: float = 1e-2,
                 kind: str = "lda", seed: int = 0) -> BeliefSubspace
```
`y` is `1` for belief-consistent, `0` for belief-violating. Raises `ValueError` if `y` has only
one class, if `X` and `y` disagree in length, or if `kind` is unknown.

```python
def normalize_margins(m: np.ndarray) -> np.ndarray
```
Zero mean, unit variance. Returns zeros for a constant input.

---

## `bmd.intervene`

```python
class ActivationSteerer:
    def __init__(self, direction: np.ndarray, layer: int, alpha: float, scale: float = 1.0)
    def apply(self, hidden: np.ndarray) -> np.ndarray     # hidden + alpha*scale*direction

class LogitBias:
    def __init__(self, vocab_vector: np.ndarray, alpha: float)
    def apply(self, logits: np.ndarray) -> np.ndarray     # logits + alpha*vocab_vector

def margin_scores(subspace: BeliefSubspace, states: np.ndarray) -> np.ndarray
```

Both `apply` methods raise `ValueError` on a width mismatch.

---

## `bmd.evaluate`

```python
def error_breakdown(pred_index: np.ndarray, items: list[ToMItem]) -> dict
    # -> {"correct": int, "world": int, "other": int}

def tom_accuracy(pred_index: np.ndarray, items: list[ToMItem]) -> float

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]
    # percentile bootstrap CI of the mean; resampling unit is the ITEM

def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]
    # Wilcoxon signed-rank -> (statistic, p_value).
    # Identical samples return (0.0, 1.0) rather than raising.
```

---

## `bmd.model`

```python
def load_model(model_id: str, device: str | None = None)
    # device=None -> mps if available, then cuda, then cpu

def extract_states(model, tokenizer, texts: list[str], layer: int,
                   device: str | None = None, batch_size: int = 8) -> np.ndarray
    # (len(texts), d) last-token hidden states

def decoder_layers(model) -> list

def score_item(model, tokenizer, prompt: str, candidates: list[str], layers: list[int],
               device: str | None = None, steerer: ActivationSteerer | None = None) -> ItemScores
```

`score_item` prefills the prompt once and reuses a **copy** of the KV cache for each candidate, so
every candidate is scored against the same context. `ItemScores.exact_tokenisation` reports
whether the joint tokenisation was used.

---

## `bmd.matrix`

```python
MECHANISMS: tuple[str, ...]
    # ("none", "prompt_oracle", "act", "margin", "logit", "margin_shuffled")

@dataclass
class MatrixConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_items: int = 200
    seed: int = 0
    layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24)
    act_alphas: tuple[float, ...] = (1.0, 3.0)
    score_alphas: tuple[float, ...] = (0.5, 1.0, 2.0)
    probe_kind: str = "lda"
    out_dir: str = "results"
    device: str | None = None
    layer_scale: float = 1.0
    skip_existing: bool = True

@dataclass
class Arm:
    mechanism: str
    layer: int | None
    alpha: float
    @property
    def label(self) -> str      # e.g. "margin-L12-a1"

def build_arms(config: MatrixConfig) -> list[Arm]
def expected_arm_count(config: MatrixConfig) -> int
def oracle_prompt(item: ToMItem) -> str
def run_matrix(config: MatrixConfig, verbose: bool = True) -> dict
```

`run_matrix` writes one JSON per arm to `<out_dir>/ablation/<label>.json` and a combined
`<out_dir>/phase0_summary.json`. It is resumable: the base pass is cached to
`<out_dir>/base_cache.npz` and completed arms are skipped.

---

## CLI

```bash
python src/run_phase0.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --n-items 200 \
  --layers 4,8,12,16,20,24 \
  --act-alphas 1.0 \
  --score-alphas 0.25,0.5,1.0,2.0,4.0 \
  --probe-kind lda \
  --out-dir results
```
