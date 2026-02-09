from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

DIVIDER = "#" * 70

FULL_FEATURES: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

MINIMAL_FEATURES: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


# -----------------------------
# Config / CLI
# -----------------------------
@dataclass(frozen=True)
class RunConfig:
    api_url: str
    csv_path: Path
    sample_size: int
    timeout_seconds: float
    require_served_by: bool
    bring_up_stack: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Send samples from future_unseen_examples.csv to /predict (full) and /predict_minimal (minimal)."
    )
    p.add_argument("--api-url", default="http://localhost:8000", help="API base URL.")
    p.add_argument("--csv-path", default=str(Path("data/future_unseen_examples.csv")), help="CSV path.")
    p.add_argument("--sample-size", type=int, default=20, help="Rows to send.")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout.")
    p.add_argument(
        "--require-served-by",
        action="store_true",
        help="Assert response includes served_by in {blue,green} and model_version.",
    )
    p.add_argument(
        "--bring-up-stack",
        action="store_true",
        help="Run `make down` + `make up` before calling the API.",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    args = build_parser().parse_args(argv)
    return RunConfig(
        api_url=str(args.api_url).rstrip("/"),
        csv_path=Path(args.csv_path),
        sample_size=int(args.sample_size),
        timeout_seconds=float(args.timeout),
        require_served_by=bool(args.require_served_by),
        bring_up_stack=bool(args.bring_up_stack),
    )


# -----------------------------
# HTTP helpers
# -----------------------------
def get_health(api_url: str, timeout_seconds: float) -> Dict[str, Any]:
    r = requests.get(f"{api_url}/health", timeout=timeout_seconds)
    r.raise_for_status()
    return r.json()


def wait_for_api_ok(api_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        try:
            body = get_health(api_url, timeout_seconds=2.0)
            if body.get("status") == "ok":
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)

    raise RuntimeError(f"API did not become healthy within {timeout_s}s. Last error: {last_err}")


def assert_api_is_healthy(api_url: str, timeout_seconds: float) -> str:
    body = get_health(api_url, timeout_seconds)
    if body.get("status") != "ok":
        raise RuntimeError(f"API healthcheck failed: {body}")
    return str(body.get("model_version", "unknown"))


def post_json(api_url: str, path: str, payload: Dict[str, Any], timeout_seconds: float) -> Tuple[int, str]:
    r = requests.post(
        f"{api_url}{path}",
        headers={"accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout_seconds,
    )
    return r.status_code, r.text


def parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"Response is not JSON: {text}") from exc


def validate_response_metadata(response_text: str) -> None:
    data = parse_json(response_text)
    served_by = str(data.get("served_by", "missing"))
    mv = str(data.get("model_version", "missing"))

    if served_by not in {"blue", "green"}:
        raise AssertionError(f"served_by must be blue|green, got {served_by}. Response: {response_text}")
    if mv in {"missing", "unknown", ""}:
        raise AssertionError(f"model_version must be present, got {mv}. Response: {response_text}")


# -----------------------------
# Docker helpers
# -----------------------------
def boot_stack_and_wait() -> None:
    subprocess.run(["make", "down"], check=False)
    subprocess.run(["make", "up"], check=True)
    wait_for_api_ok("http://localhost:8000", timeout_s=60)


# -----------------------------
# Data loading
# -----------------------------
def load_samples(csv_path: Path, sample_size: int) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    df = pd.read_csv(csv_path, dtype={"zipcode": str})
    n = min(sample_size, len(df))
    df_sample = df.sample(n, random_state=42)

    # keep your debug output
    print(df_sample.columns)

    records: List[Dict[str, Any]] = []
    for row in df_sample.to_dict(orient="records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        records.append(clean)
    return records


# -----------------------------
# Payload builders (STRICT)
# -----------------------------
def _pick(case: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    payload = {k: case.get(k) for k in keys}

    # enforce zipcode as str (pydantic expects str)
    if payload.get("zipcode") is not None:
        payload["zipcode"] = str(payload["zipcode"])

    return payload


def build_payload(endpoint: str, case: Dict[str, Any]) -> Dict[str, Any]:
    if endpoint == "/predict":
        # ✅ must match PredictionRequestFull exactly
        return _pick(case, FULL_FEATURES)

    if endpoint == "/predict_minimal":
        # ✅ must match PredictionRequestMinimal exactly
        return _pick(case, MINIMAL_FEATURES)

    raise ValueError(f"Unknown endpoint: {endpoint}")


# -----------------------------
# Output
# -----------------------------
def print_header(endpoint: str, cfg: RunConfig, model_version: str, cases: List[Dict[str, Any]]) -> None:
    print(DIVIDER)
    print(f"Endpoint: {endpoint}")
    print(f"API: {cfg.api_url}")
    print(f"Model version: {model_version}")
    print(f"CSV: {cfg.csv_path.resolve()}")
    print(f"Sample size: {len(cases)}")
    print(DIVIDER)


def print_success(case: Dict[str, Any], payload: Dict[str, Any], response_text: str) -> None:
    print(DIVIDER)
    print(f"Success! Case: {case}")
    print(f"Sent payload: {payload}")
    print(f"Response: {response_text}")
    print(DIVIDER)


def print_failure(case: Dict[str, Any], payload: Dict[str, Any], status_code: int, response_text: str) -> None:
    print(DIVIDER)
    print(f"Error! Case: {case}")
    print(f"Sent payload: {payload}")
    print(f"Status code: {status_code}")
    print(f"Response: {response_text}")
    print(DIVIDER)


def print_summary(successes: int, failures: int) -> None:
    print("\n--- Summary ---")
    print(f"Successes: {successes}")
    print(f"Failures:  {failures}")


# -----------------------------
# Runner
# -----------------------------
def run_endpoint(cfg: RunConfig, endpoint: str, cases: List[Dict[str, Any]], model_version: str) -> int:
    print_header(endpoint, cfg, model_version, cases)

    successes = 0
    failures = 0

    for case in cases:
        payload = build_payload(endpoint, case)
        status_code, response_text = post_json(cfg.api_url, endpoint, payload, cfg.timeout_seconds)

        if 200 <= status_code < 300:
            successes += 1
            if cfg.require_served_by:
                validate_response_metadata(response_text)
            print_success(case, payload, response_text)
        else:
            failures += 1
            print_failure(case, payload, status_code, response_text)

    print_summary(successes, failures)
    return 0 if failures == 0 else 1


def run_both_endpoints(cfg: RunConfig) -> int:
    if cfg.bring_up_stack:
        boot_stack_and_wait()
    else:
        wait_for_api_ok(cfg.api_url, timeout_s=60)

    model_version = assert_api_is_healthy(cfg.api_url, cfg.timeout_seconds)
    cases = load_samples(cfg.csv_path, cfg.sample_size)

    rc1 = run_endpoint(cfg, "/predict", cases, model_version)
    rc2 = run_endpoint(cfg, "/predict_minimal", cases, model_version)

    return 0 if (rc1 == 0 and rc2 == 0) else 1


# -----------------------------
# CLI entrypoint
# -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    return run_both_endpoints(cfg)


# -----------------------------
# Pytest: single test running both endpoints
# -----------------------------
def test_live_unseen_examples_both_endpoints():
    boot_stack_and_wait()

    cfg = RunConfig(
        api_url="http://localhost:8000",
        csv_path=Path("data/future_unseen_examples.csv"),
        sample_size=20,
        timeout_seconds=10.0,
        require_served_by=True,
        bring_up_stack=False,
    )

    assert run_both_endpoints(cfg) == 0


if __name__ == "__main__":
    raise SystemExit(main())