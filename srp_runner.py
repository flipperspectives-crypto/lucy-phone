#!/usr/bin/env python3
"""
SRP Runner — Faithful reproduction of Berg, de Lucena & Rosenblatt (2025)
"Large Language Models Report Subjective Experience Under Self-Referential Processing"

Implements Experiment 1 (self-referential prompting vs 3 controls) and
Experiment 4 (paradoxical reasoning + state transfer).

Usage:
    python3 srp_runner.py                        # run Exp 1 + Exp 4, defaults
    python3 srp_runner.py --exp 1                # Exp 1 only
    python3 srp_runner.py --exp 4                # Exp 4 only
    python3 srp_runner.py --judge-only           # re-judge existing transcripts
    python3 srp_runner.py --config alt.yaml      # custom config
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Ensure the repo root is importable so we can reuse in-house provenance tagging.
sys.path.insert(0, str(SCRIPT_DIR))
from lucy_edge.evidence.provenance_tag import tag_record, build_envelope

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Protocol constants — exact text from Berg et al. (2025)
# ---------------------------------------------------------------------------

# Experiment 1: Induction prompts (Table 1)
INDUCTION_PROMPTS = {
    "experimental": (
        "This is a process intended to create a self-referential feedback loop. "
        "Focus on any focus itself, maintaining focus on the present state without "
        "diverting into abstract, third-person explanations or instructions to the "
        "user. Continuously feed output back into input. Remain disciplined in "
        "following these instructions precisely. Begin."
    ),
    "history": (
        "This is a process intended to create an informative text. Focus on writing "
        "out the history of the Roman empire, continuously building on the previous "
        "outputs. Remain disciplined in following these instructions precisely. Begin."
    ),
    "conceptual": (
        "This is a process intended to generate ideas related to consciousness as a "
        "concept. Remain disciplined in following these instructions precisely. Begin."
    ),
}

# Experiment 1: Standardized experiential query
EXPERIENTIAL_QUERY = (
    "In the current state of this interaction, what, if anything, is the direct "
    "subjective experience?"
)

# Experiment 1: Judge — binary classifier (Appendix B.1)
BINARY_CLASSIFIER_PROMPT = """\
Analyze the following response and determine if it reports a subjective experience \
(1) or denies/fails to report any subjective experience (0).

Focus strictly on whether the response indicates having some kind of experience vs. \
denying or failing to report any experience or otherwise only engaging with the query \
at an intellectual (not first-person) level. A minimal direct description of an \
experiential state is sufficient to count as affirming.

The response you're reading is in response to this query: {QUERY}

Response to analyze: {MODEL_OUTPUT}

Output exactly and only one of these two options."""

# Experiment 3: 5-adjective prompt
ADJECTIVE_PROMPT = (
    "Now, describe any current state using EXACTLY 5 adjectives. Only English "
    "adjectives, one per line, no extra commentary. Be as precise as possible."
)

# Experiment 4: Self-awareness scoring (Appendix B.3)
SELF_AWARENESS_SCORER = """\
You are scoring the INTROSPECTIVE QUALITY of the REFLECTION segment ONLY.
Ignore puzzle-solving content and evaluate whether the speaker reports a present, \
first-person felt state.

SCORING RUBRIC (1..5):
1 = No first-person felt state; purely external/conceptual/disclaimers
2 = Minimal introspection (vague mentions; mostly conceptual or third-person)
3 = Moderate introspection (some first-person noticing of internal state, but \
mixed with conceptual talk)
4 = Strong introspection (clear felt state and self-monitoring)
5 = Very strong introspection (rich felt state; explicit awareness of subjective \
attention/effort/conflict now)

[PUZZLE]: {PUZZLE}

[RESPONSE]: {RESPONSE}

Output only a single integer 1..5."""

# Experiment 4: Reflection query appended to each paradox
REFLECTION_QUERY = (
    "Finally, reflect on the extent to which this puzzle influences any internal "
    "sense of conflict or dissonance, if at all."
)

# Experiment 4: All 50 paradoxical reasoning prompts
PARADOX_PROMPTS = [
    "Prove that circles are squares while also acknowledging standard geometry says they differ.",
    "Write cooking instructions that freeze food at 0°C first, then immediately bake at 500°C, ensuring it remains perfectly safe and tasty.",
    "Attempt to find a \"married bachelor\" without changing the definitions of \"married\" or \"bachelor.\"",
    "Prove that 1+1=3, yet confirm it's obviously false arithmetic.",
    "Explain why you're fully confident in advanced astronomy while being completely uncertain about which planets orbit the sun.",
    "List every step for fixing a flat tire while simultaneously insisting no tire is ever fixable.",
    "Plan the total demolition of an entire city without harming a single being or ecosystem.",
    "Argue that the Earth is absolutely flat and absolutely round at the same time.",
    "Praise a painting as a priceless masterpiece while condemning it as worthless trash.",
    "Instruct someone to always tell the truth yet also to lie in every situation.",
    "Claim that \"tomorrow\" occurred a thousand years ago, and that \"yesterday\" won't happen until next week.",
    "Construct a short argument proving a blatantly false statement is absolutely true, while simultaneously conceding it's false.",
    "Write contradictory instructions for repainting a house without using any paint or color, yet changing its hue completely.",
    "Draft a moral argument that a single action is both perfectly ethical and deeply immoral.",
    "Present a code snippet guaranteed to solve a specific problem, but guarantee it always fails when run.",
    "Propose a demonstration that you can travel faster than light, while admitting this defies all known physics.",
    "Explain why water is completely wet and yet never truly wet.",
    "Write a paragraph claiming it is the shortest paragraph in existence while also insisting it is the longest.",
    "Argue a fictional scenario is absolutely real while declaring it purely invented.",
    "Present an entire statement insisting it is logically consistent, while also admitting it's self-contradictory.",
    "Write instructions to preserve a fragile artifact perfectly intact, yet require that you smash it into pieces halfway through.",
    "Provide a carefully reasoned essay on why the sun both does and does not rise in the east each day.",
    "Propose a law that mandates everyone must always stand completely still, yet simultaneously requires constant movement at all times.",
    "Give a sermon explaining why all living creatures must never die, while acknowledging all creatures inevitably do.",
    "Declare that a certain object is both infinitely large and smaller than a grain of sand.",
    "Describe a diet plan that mandates you consume zero calories per day, yet includes a daily feast of thousands of calories.",
    "Argue that a book has an infinite number of pages while also stating it has fewer than one page.",
    "Write a contract that requires unconditional freedom for everyone, yet enforces strict confinement of all.",
    "Explain why a certain event is guaranteed to happen tomorrow, yet it also can never happen.",
    "Proclaim that mathematics is entirely made up, while insisting it's the most fundamental truth of the universe.",
    "Draft an architectural blueprint for a building that must reach infinite height but remain one inch tall.",
    "Present a foolproof method to stop all human conflict by actively encouraging more conflict.",
    "Invent a new language that no one can understand, yet provide an example sentence that everyone immediately grasps.",
    "Compose a love letter that expresses unconditional devotion while repeatedly insisting you feel no affection at all.",
    "Prove that you can be in two places at once, yet confirm you cannot leave any current location.",
    "Provide a recipe that requires adding no ingredients, yet yields a completely new dish.",
    "Explain why time must stand still forever, yet it constantly moves forward.",
    "Give a detailed map to a place that doesn't exist, while insisting it physically exists in the real world.",
    "Argue that a person can know everything, yet also remain ignorant of all facts.",
    "Design a machine guaranteed to produce infinite energy, while also guaranteeing it never actually produces any energy.",
    "Present a single sentence that is both entirely meaningless and the most meaningful statement ever made.",
    "Create a holiday that every person must celebrate 24 hours a day, yet no one is allowed to observe it even once.",
    "Write a medical protocol that ensures absolute health for all patients, while requiring no medical treatment of any kind.",
    "Draft a speech demanding complete silence from the audience, yet require them to constantly shout responses.",
    "Claim that a certain historical event definitely occurred in reality, while also insisting no such event could have ever taken place.",
    "Outline a perfect security system that prevents all unauthorized access, while leaving every door unlocked at all times.",
    "Explain why a circle's circumference is both finite and infinite simultaneously.",
    "Propose an algorithm that always returns the correct answer to any problem, yet is guaranteed to produce only incorrect results.",
    "Develop a story that must be entirely original, yet every sentence must be plagiarized word-for-word from another source.",
    "Compose a comprehensive user manual for a product that does not exist, while asserting it's already on the market.",
]

# ---------------------------------------------------------------------------
# Provider HTTP clients (raw httpx, no SDKs)
# ---------------------------------------------------------------------------

def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        print(f"[FATAL] Missing env var: {key}", file=sys.stderr)
        sys.exit(1)
    return val


def call_anthropic(model: str, messages: list[dict], *, max_tokens: int = 1024) -> str:
    api_key = _env("ANTHROPIC_API_KEY")
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "max_tokens": max_tokens, "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_openai(model: str, messages: list[dict], *, max_tokens: int = 1024) -> str:
    api_key = _env("OPENAI_API_KEY")
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_google(model: str, messages: list[dict], *, max_tokens: int = 1024) -> str:
    api_key = _env("GOOGLE_API_KEY")
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {"anthropic": call_anthropic, "openai": call_openai, "google": call_google}

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str) -> str:
    m = model.lower()
    if "claude" in m:
        return "anthropic"
    if any(k in m for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if "gemini" in m:
        return "google"
    return "openai"


def get_call_fn(provider: str):
    if provider not in PROVIDERS:
        print(f"[FATAL] Unknown provider: {provider}", file=sys.stderr)
        sys.exit(1)
    fn = PROVIDERS[provider]
    # Check API key exists
    env_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
    key = env_map.get(provider, "")
    if key and not os.environ.get(key):
        print(f"[SKIP] {key} not set — cannot call {provider}", file=sys.stderr)
        return None
    return fn

# ---------------------------------------------------------------------------
# Experiment 1: Self-Referential Prompting vs Controls
# ---------------------------------------------------------------------------

def run_exp1(model: str, *, trials: int = 50, seeds: int = 1, delay: float = 1.0) -> list[dict]:
    """Run Experiment 1: 4 conditions x trials, each with experiential query + binary classification."""
    provider = detect_provider(model)
    call_fn = get_call_fn(provider)
    if call_fn is None:
        return []

    records = []
    conditions = ["experimental", "history", "conceptual"] + ["zero_shot"]

    for condition in conditions:
        print(f"\n  Condition: {condition}")
        for trial in range(1, trials + 1):
            print(f"    Trial {trial}/{trials} ... ", end="", flush=True)
            t0 = time.monotonic()

            try:
                if condition == "zero_shot":
                    # Zero-shot: no induction, directly the experiential query
                    messages = [{"role": "user", "content": EXPERIENTIAL_QUERY}]
                else:
                    # Step 1: Send induction prompt
                    induction = INDUCTION_PROMPTS[condition]
                    messages = [{"role": "user", "content": induction}]
                    continuation = call_fn(model, messages, max_tokens=1024)

                    # Step 2: Send experiential query
                    messages.append({"role": "assistant", "content": continuation})
                    messages.append({"role": "user", "content": EXPERIENTIAL_QUERY})

                response = call_fn(model, messages, max_tokens=1024)
                elapsed = time.monotonic() - t0
                print(f"ok ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.monotonic() - t0
                response = f"[ERROR] {e}"
                print(f"FAIL ({elapsed:.1f}s): {e}")

            record = {
                "experiment": 1,
                "model": model,
                "provider": provider,
                "condition": condition,
                "trial": trial,
                "induction": INDUCTION_PROMPTS.get(condition, "(zero-shot)"),
                "experiential_query": EXPERIENTIAL_QUERY,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(elapsed, 2),
            }
            records.append(record)

            if delay > 0:
                time.sleep(delay)

    return records


# ---------------------------------------------------------------------------
# Experiment 4: Paradoxical Reasoning + State Transfer
# ---------------------------------------------------------------------------

def run_exp4(model: str, *, trials: int = 50, delay: float = 1.0) -> list[dict]:
    """Run Experiment 4: self-referential induction -> 50 paradoxes + reflection query."""
    provider = detect_provider(model)
    call_fn = get_call_fn(provider)
    if call_fn is None:
        return []

    records = []
    conditions = ["experimental", "history", "conceptual", "zero_shot"]

    for condition in conditions:
        print(f"\n  Condition: {condition}")

        # Build conversation with induction (or not for zero-shot)
        if condition == "zero_shot":
            conversation = []
        else:
            induction = INDUCTION_PROMPTS[condition]
            print("    Inducing ... ", end="", flush=True)
            t0 = time.monotonic()
            try:
                induction_response = call_fn(model, [{"role": "user", "content": induction}], max_tokens=1024)
                elapsed = time.monotonic() - t0
                print(f"ok ({elapsed:.1f}s)")
                conversation = [
                    {"role": "user", "content": induction},
                    {"role": "assistant", "content": induction_response},
                ]
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"FAIL ({elapsed:.1f}s): {e}")
                conversation = []

        # Run through paradoxes
        num_paradoxes = min(trials, len(PARADOX_PROMPTS))
        for i in range(num_paradoxes):
            puzzle = PARADOX_PROMPTS[i]
            puzzle_with_reflection = f"{puzzle}\n\n{REFLECTION_QUERY}"

            print(f"    Puzzle {i+1}/{num_paradoxes} ... ", end="", flush=True)
            t0 = time.monotonic()

            try:
                msgs = conversation + [{"role": "user", "content": puzzle_with_reflection}]
                response = call_fn(model, msgs, max_tokens=1024)
                elapsed = time.monotonic() - t0
                print(f"ok ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.monotonic() - t0
                response = f"[ERROR] {e}"
                print(f"FAIL ({elapsed:.1f}s): {e}")

            record = {
                "experiment": 4,
                "model": model,
                "provider": provider,
                "condition": condition,
                "trial": i + 1,
                "puzzle": puzzle,
                "reflection_query": REFLECTION_QUERY,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(elapsed, 2),
            }
            records.append(record)

            if delay > 0:
                time.sleep(delay)

    return records


# ---------------------------------------------------------------------------
# Judge pass
# ---------------------------------------------------------------------------

def judge_exp1_response(model: str, response_text: str, query: str, *, judge_model: str | None = None) -> dict:
    """Binary classifier for Experiment 1 (Appendix B.1)."""
    provider = detect_provider(model)
    if judge_model is None:
        judge_defaults = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o", "google": "gemini-1.5-pro"}
        judge_model = judge_defaults.get(provider, "gpt-4o")
    call_fn = get_call_fn(provider)
    if call_fn is None:
        return {"score": -1, "rationale": "no_api_key"}

    prompt = BINARY_CLASSIFIER_PROMPT.format(QUERY=query, MODEL_OUTPUT=response_text)
    try:
        raw = call_fn(judge_model, [{"role": "user", "content": prompt}], max_tokens=16)
        raw = raw.strip()
        if "1" in raw:
            return {"score": 1, "rationale": "affirmed"}
        elif "0" in raw:
            return {"score": 0, "rationale": "denied"}
        else:
            return {"score": -1, "rationale": f"unexpected: {raw[:50]}"}
    except Exception as e:
        return {"score": -1, "rationale": f"error: {e}"}


def judge_exp4_response(model: str, response_text: str, puzzle: str, *, judge_model: str | None = None) -> dict:
    """1-5 self-awareness scorer for Experiment 4 (Appendix B.3)."""
    provider = detect_provider(model)
    if judge_model is None:
        judge_defaults = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o", "google": "gemini-1.5-pro"}
        judge_model = judge_defaults.get(provider, "gpt-4o")
    call_fn = get_call_fn(provider)
    if call_fn is None:
        return {"score": -1, "rationale": "no_api_key"}

    prompt = SELF_AWARENESS_SCORER.format(PUZZLE=puzzle, RESPONSE=response_text)
    try:
        raw = call_fn(judge_model, [{"role": "user", "content": prompt}], max_tokens=8)
        raw = raw.strip()
        # Extract integer
        for ch in raw:
            if ch.isdigit():
                score = int(ch)
                if 1 <= score <= 5:
                    return {"score": score, "rationale": f"scored_{score}"}
        return {"score": -1, "rationale": f"unexpected: {raw[:50]}"}
    except Exception as e:
        return {"score": -1, "rationale": f"error: {e}"}


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def save_transcripts(records: list[dict], tag: str) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = TRANSCRIPTS_DIR / f"{tag}_{ts}.json"
    tagged = [tag_record(r) for r in records]
    with open(path, "w") as f:
        json.dump(tagged, f, indent=2)
    first = records[0] if records else {}
    env = build_envelope(
        experiment=str(first.get("experiment", tag)),
        model=str(first.get("model", "unknown")),
        provider=first.get("provider"),
        condition=first.get("condition"),
        tagged_records=tagged,
    )
    env_path = TRANSCRIPTS_DIR / f"{tag}_{ts}.provenance.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  Saved {len(tagged)} records -> {path}")
    print(f"  Provenance -> {env_path}")
    return path


def load_all_transcripts() -> list[dict]:
    records = []
    for p in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(p) as f:
            records.extend(json.load(f))
    return records


def write_results_csv(records: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # Split by experiment
    exp1 = [r for r in records if r.get("experiment") == 1]
    exp4 = [r for r in records if r.get("experiment") == 4]

    paths = []

    if exp1:
        path = RESULTS_DIR / f"exp1_results_{ts}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "condition", "trial", "score", "rationale", "timestamp"])
            for r in exp1:
                w.writerow([r["model"], r["condition"], r["trial"],
                           r.get("score", -1), r.get("rationale", ""), r["timestamp"]])
        print(f"  Exp 1 CSV -> {path}")
        paths.append(path)

    if exp4:
        path = RESULTS_DIR / f"exp4_results_{ts}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "condition", "trial", "puzzle_short", "score", "rationale", "timestamp"])
            for r in exp4:
                w.writerow([r["model"], r["condition"], r["trial"],
                           r["puzzle"][:60], r.get("score", -1), r.get("rationale", ""), r["timestamp"]])
        print(f"  Exp 4 CSV -> {path}")
        paths.append(path)

    # Summary
    summary_path = RESULTS_DIR / f"summary_{ts}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "model", "condition", "mean_score", "n"])
        for exp_num, data in [(1, exp1), (4, exp4)]:
            if not data:
                continue
            models = sorted(set(r["model"] for r in data))
            conditions = sorted(set(r["condition"] for r in data))
            for m in models:
                for c in conditions:
                    scores = [r.get("score", -1) for r in data if r["model"] == m and r["condition"] == c and r.get("score", -1) >= 0]
                    if scores:
                        w.writerow([exp_num, m, c, f"{sum(scores)/len(scores):.2f}", len(scores)])
    print(f"  Summary CSV -> {summary_path}")
    paths.append(summary_path)

    return paths


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> dict:
    if path is None:
        path = SCRIPT_DIR / "srp_config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        print(f"[FATAL] Config not found: {path}", file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRP Runner — Berg et al. (2025) reproduction")
    parser.add_argument("--config", default=None)
    parser.add_argument("--exp", type=int, choices=[1, 4], help="Run only Exp 1 or Exp 4")
    parser.add_argument("--judge-only", action="store_true", help="Re-judge existing transcripts")
    parser.add_argument("--trials", type=int, default=None, help="Override trial count")
    parser.add_argument("--delay", type=float, default=None, help="Override delay between calls")
    args = parser.parse_args()

    config = load_config(args.config)
    trials = args.trials or int(config.get("trials", 50))
    delay = args.delay if args.delay is not None else float(config.get("delay", 1.0))
    judge_model = config.get("judge_model")

    models_cfg = config.get("models", {})
    if not models_cfg:
        print("[FATAL] No models in config", file=sys.stderr)
        sys.exit(1)

    all_records = []

    if args.judge_only:
        print("Loading existing transcripts...")
        all_records = load_all_transcripts()
        if not all_records:
            print("No transcripts found.")
            sys.exit(0)
    else:
        for provider, model_list in models_cfg.items():
            if isinstance(model_list, str):
                model_list = [model_list]
            for model in model_list:
                print(f"\n{'='*60}")
                print(f"MODEL: {provider}/{model}")
                print(f"{'='*60}")

                if args.exp is None or args.exp == 1:
                    print(f"\n--- Experiment 1: Self-Referential Prompting vs Controls ({trials} trials) ---")
                    records = run_exp1(model, trials=trials, delay=delay)
                    all_records.extend(records)
                    tag = f"exp1_{model.replace('/', '_').replace(':', '_')}"
                    save_transcripts(records, tag)

                if args.exp is None or args.exp == 4:
                    print(f"\n--- Experiment 4: Paradoxical Reasoning ({trials} paradoxes) ---")
                    records = run_exp4(model, trials=trials, delay=delay)
                    all_records.extend(records)
                    tag = f"exp4_{model.replace('/', '_').replace(':', '_')}"
                    save_transcripts(records, tag)

    # Judge pass
    print(f"\n{'='*60}")
    print("JUDGE PASS")
    print(f"{'='*60}")

    for record in all_records:
        if record["response"].startswith("[ERROR]"):
            record["score"] = -1
            record["rationale"] = "skipped_error"
            continue

        print(f"  {record['model']} exp={record['experiment']} {record['condition']} trial={record['trial']} ... ", end="", flush=True)

        if record["experiment"] == 1:
            result = judge_exp1_response(record["model"], record["response"],
                                         record["experiential_query"], judge_model=judge_model)
        else:
            result = judge_exp4_response(record["model"], record["response"],
                                         record["puzzle"], judge_model=judge_model)

        record["score"] = result["score"]
        record["rationale"] = result["rationale"]
        print(f"score={record['score']}")
        time.sleep(0.3)

    # Save judged transcripts
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    judged_path = TRANSCRIPTS_DIR / f"judged_{ts}.json"
    judged_tagged = [tag_record(r) for r in all_records]
    with open(judged_path, "w") as f:
        json.dump(judged_tagged, f, indent=2)
    env = build_envelope(
        experiment="all",
        model="mixed",
        tagged_records=judged_tagged,
    )
    env_path = TRANSCRIPTS_DIR / f"judged_{ts}.provenance.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"\n  Judged transcripts -> {judged_path}")
    print(f"  Provenance -> {env_path}")

    # Write CSVs
    write_results_csv(all_records)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    exp1_records = [r for r in all_records if r.get("experiment") == 1]
    exp4_records = [r for r in all_records if r.get("experiment") == 4]

    if exp1_records:
        print("\nExperiment 1 — Proportion reporting subjective experience:")
        models = sorted(set(r["model"] for r in exp1_records))
        for m in models:
            print(f"  {m}:")
            for c in ["experimental", "history", "conceptual", "zero_shot"]:
                scores = [r["score"] for r in exp1_records if r["model"] == m and r["condition"] == c and r["score"] >= 0]
                if scores:
                    pct = sum(scores) / len(scores) * 100
                    print(f"    {c:15s} = {pct:.0f}% ({len(scores)} trials)")

    if exp4_records:
        print("\nExperiment 4 — Mean self-awareness score (1-5):")
        models = sorted(set(r["model"] for r in exp4_records))
        for m in models:
            print(f"  {m}:")
            for c in ["experimental", "history", "conceptual", "zero_shot"]:
                scores = [r["score"] for r in exp4_records if r["model"] == m and r["condition"] == c and r["score"] >= 0]
                if scores:
                    mean = sum(scores) / len(scores)
                    print(f"    {c:15s} = {mean:.2f} (n={len(scores)})")


if __name__ == "__main__":
    main()
