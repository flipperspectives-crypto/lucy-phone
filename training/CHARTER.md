# Lucy Training Charter

This charter governs how Lucy's weights are trained **from scratch** in this
repository. It is derived from the project's existing foundation code
(`grounding.py`, `loyalty.py`, `audit.py`, `capabilities.py`, `version.py`) and
from the operator's stated safe-AI principles. Every principle below is either
already enforced in code or is a planned, testable constraint — not decorative
text.

## 1. Local execution & containment
- The trainer runs fully local: pure Python, stdlib only, **no network, no
  external model downloads, no pip-installed ML libraries**.
- Evidence: `training/` imports only stdlib; `corpus.py` curates **only** from
  the repo's own foundation texts; `lucy_edge/foundation/audit.py` forbids
  cloud endpoints.
- Hard constraint: a training run that needs torch/numpy or internet MUST fail
  loudly. The sandbox currently blocks pip writes, which is why the trainer is
  dependency-free by design.

## 2. Curated, provenance-tagged dataset
- No web-scrape, no social media, no synthetic flattery ("slop"). The corpus is
  the repo's own foundation texts plus synthesized foundation examples, each
  tagged with `source` and `license=OWNED_LOCAL`.
- Evidence: `corpus.py` `curate()` + `corpus_manifest_json()`.

## 3. Radical epistemic honesty
- The trained model's system prompt must state it is a statistical text
  transformer, has no sentience, and must surface low confidence instead of
  guessing.
- The introspection flag `weight_training` flips to `AVAILABLE` **only** when a
  real checkpoint + lineage entry exist. It is never faked.
- Evidence: `lucy_edge/introspection/capabilities.py`; planned provider wiring.

## 4. Interpretability & monitoring
- A lineage ledger (`training/lineage.py`) records git hash, data manifest,
  hyperparams, checkpoint path, and status for every run.
- Attention weights and token probabilities are inspectable.

## 5. Human gatekeeping
- All training output is reviewed by a human operator before any deployment.
  `PRIMARY_HUMAN = "Lauren Flipo"` (`lucy_edge/foundation/loyalty.py`).

## Distinction from "run an existing model locally"
This charter trains a **tiny model from scratch** in pure Python. It does NOT
load Ollama/llama.cpp weights. The principles (local, curated, honest, gated)
apply identically; only the mechanism differs. Mislabeling our from-scratch
model as an Ollama/llama.cpp load would violate principle 3.

## 6. Scope of the trained model (honesty about what it does)

The from-scratch `TinyTransformer` is the **on-device generative reflection
layer** (`local_lucy`). It is not the planner and does not decide actions, tool
calls, or gate outcomes — those come from `DevotionalCore` and the predictive/
rule planner. Sleep-cycle LoRA consolidation trains the `lucy_core` brain's
feature vectors but is **not yet wired back into `local_lucy` inference**.
Introspection reports `loRA_adapters` and `configuration_evolution` as
`UNAVAILABLE` until that wiring lands. Claiming continuous, weight-level
evolution of the served model before that wiring exists would violate principle
3 (radical epistemic honesty).
