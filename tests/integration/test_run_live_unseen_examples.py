from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


DIVIDER = "#" * 70


@dataclass(frozen=True)
class RunConfig:
    api_url: str
    csv_path: Path
    sample_size: int
    timeout_seconds: float
    require_served_by: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Send samples from future_unseen_examples.csv to the live /predict endpoint."
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--csv-path",
        default=str(Path("app/data/future_unseen_examples.csv")),
        help="Path to future_unseen_examples.csv (default: app/data/future_unseen_examples.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of rows to send (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--require-served-by",
        action="store_true",
        help="If set, each successful /predict response must include served_by in {blue, green}.",
    )
    args = parser.parse_args()

    return RunConfig(
        api_url=args.api_url.rstrip("/"),
        csv_path=Path(args.csv_path),
        sample_size=args.sample_size,
        timeout_seconds=args.timeout,
        require_served_by=bool(args.require_served_by),
    )


def assert_api_is_healthy(api_url: str, timeout_seconds: float) -> str:
    url = f"{api_url}/health"
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    body = resp.json()

    if body.get("status") != "ok":
        raise RuntimeError(f"API healthcheck failed: {body}")

    return str(body.get("model_version", "unknown"))


def wait_for_api_ok(api_url: str, timeout_s: float = 45.0) -> None:
    """
    Wait until GET /health returns {"status":"ok"} (helps avoid flaky CI race conditions
    right after `make up`).
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url.rstrip('/')}/health", timeout=2)
            if r.status_code == 200:
                try:
                    body = r.json()
                except Exception:
                    body = {}
                if body.get("status") == "ok":
                    return
        except Exception as e:
            last_err = e

        time.sleep(0.5)

    raise RuntimeError(f"API did not become healthy within {timeout_s}s. Last error: {last_err}")


def load_samples(csv_path: Path, sample_size: int) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, dtype={"zipcode": str})

    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    sample_size = min(sample_size, len(df))
    df_sample = df.sample(sample_size, random_state=42)

    records: List[Dict[str, Any]] = []
    for row in df_sample.to_dict(orient="records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        records.append(clean)

    return records


def post_predict(api_url: str, payload: Dict[str, Any], timeout_seconds: float) -> Tuple[int, str]:
    url = f"{api_url}/predict"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_seconds)
    return resp.status_code, resp.text


def main() -> int:
    cfg = parse_args()

    try:
        model_version = assert_api_is_healthy(cfg.api_url, cfg.timeout_seconds)
    except Exception as exc:
        print(f"{DIVIDER}\nERROR: API is not healthy\n{exc}\n{DIVIDER}")
        return 2

    try:
        cases = load_samples(cfg.csv_path, cfg.sample_size)
    except Exception as exc:
        print(f"{DIVIDER}\nERROR: Failed to load CSV\n{exc}\n{DIVIDER}")
        return 3

    print(DIVIDER)
    print(f"API: {cfg.api_url}")
    print(f"Model version: {model_version}")
    print(f"CSV: {cfg.csv_path.resolve()}")
    print(f"Sample size: {len(cases)}")
    print(DIVIDER)

    successes = 0
    failures = 0

    for case in cases:
        status_code, response_text = post_predict(cfg.api_url, case, cfg.timeout_seconds)

        if 200 <= status_code < 300:
            successes += 1

            if cfg.require_served_by:
                try:
                    data = json.loads(response_text)
                except Exception as exc:
                    raise RuntimeError(f"Response is not JSON: {response_text}") from exc

                served_by = str(data.get("served_by", "missing"))
                mv = str(data.get("model_version", "missing"))
                if served_by not in {"blue", "green"}:
                    raise AssertionError(
                        f"served_by must be blue|green in blue/green mode, got {served_by}. "
                        f"Response was: {response_text}"
                    )
                if mv in {"missing", "unknown", ""}:
                    raise AssertionError(
                        f"model_version must be present in response, got {mv}. "
                        f"Response was: {response_text}"
                    )

            print(DIVIDER)
            print(f"Success! Case: {case}")
            print(f"Response: {response_text}")
            print(DIVIDER)
        else:
            failures += 1
            print(DIVIDER)
            print(f"Error! Case: {case}")
            print(f"Status code: {status_code}")
            print(f"Response: {response_text}")
            print(DIVIDER)

    print("\n--- Summary ---")
    print(f"Successes: {successes}")
    print(f"Failures:  {failures}")

    return 0 if failures == 0 else 1


def test_run_live_unseen_examples():
    subprocess.run(["make", "down"], check=False)  # don't fail if nothing is running
    subprocess.run(["make", "up"], check=True)

    # wait until the API is truly ready (avoids ConnectionResetError / 502 right after startup)
    wait_for_api_ok("http://localhost:8000", timeout_s=60)

    sys.argv = [
        "test_run_live_unseen_examples.py",
        "--api-url",
        "http://localhost:8000",
        "--csv-path",
        "data/future_unseen_examples.csv",
        "--sample-size",
        "20",
        "--timeout",
        "10",
        "--require-served-by",
    ]
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())