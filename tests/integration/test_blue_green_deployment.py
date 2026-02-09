import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Config
# -----------------------------
COMPOSE_FILE = "deploy/docker-compose.bluegreen.yml"

BASE_URL = "http://localhost:8000"
HEALTH_PATH = "/health"

N_HITS = 20
MIN_RATIO = 0.90
RELOAD_SETTLE_SECONDS = 0.25

NGINX_SERVICE = "nginx"
BLUE_SERVICE = "api_blue"
GREEN_SERVICE = "api_green"
UPSTREAM_CONF = "deploy/nginx/conf.d/upstream.conf"

CID_RE = re.compile(r"\b[a-f0-9]{12,64}\b")


@dataclass(frozen=True)
class CurlResult:
    status_code: int
    body: str


# -----------------------------
# Shell helpers
# -----------------------------
def run(cmd: str, check: bool = True) -> str:
    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")
    return p.stdout


def compose(cmd: str, check: bool = True) -> str:
    return run(f'docker compose -f "{COMPOSE_FILE}" {cmd}', check=check)


# -----------------------------
# Docker helpers
# -----------------------------
def get_container_id(service: str) -> str:
    out = compose(f"ps -q {service}", check=False).strip()
    m = CID_RE.search(out)
    if not m:
        raise RuntimeError(
            f"Container for service '{service}' not found. Is the stack up?\nRaw output:\n{out}"
        )
    return m.group(0)


def get_ip(service: str) -> str:
    cid = get_container_id(service)
    out = run(
        "docker inspect -f "
        f"'{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' "
        f'"{cid}"'
    ).strip()
    if not out:
        raise RuntimeError(f"Could not get IP for {service}")
    return out


# -----------------------------
# Nginx helpers
# -----------------------------
def nginx_reload() -> None:
    cid = get_container_id(NGINX_SERVICE)
    run(f'docker exec -t "{cid}" nginx -t >/dev/null')
    run(f'docker exec -t "{cid}" nginx -s reload >/dev/null')
    time.sleep(RELOAD_SETTLE_SECONDS)


def write_upstream_conf(service: str) -> None:
    conf = f"upstream api_upstream {{ server {service}:8000; }}\n"
    run('mkdir -p "$(dirname \\"' + UPSTREAM_CONF + '\\")"')
    run(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"Path('{UPSTREAM_CONF}').write_text({json.dumps(conf)})\n"
        f"print('Wrote {UPSTREAM_CONF} -> {service}')\n"
        "PY"
    )


def set_upstream(service: str) -> None:
    write_upstream_conf(service)
    nginx_reload()


# -----------------------------
# HTTP helpers
# -----------------------------
def curl_health() -> CurlResult:
    """
    Use short connect + total timeouts so tests never hang.
    Returns (status_code, body). If parsing fails, returns status_code=0.
    """
    out = run(
        f'curl -s --connect-timeout 1 --max-time 2 -w "\\n%{{http_code}}" "{BASE_URL}{HEALTH_PATH}"',
        check=False,
    )
    if "\n" not in out:
        return CurlResult(status_code=0, body=out)

    body, code = out.rsplit("\n", 1)
    try:
        return CurlResult(status_code=int(code.strip()), body=body.strip())
    except ValueError:
        return CurlResult(status_code=0, body=out)


def wait_for_health_ok(timeout_s: float = 30.0, sleep_s: float = 0.25) -> None:
    """
    NGINX can return transient 502/000 immediately after upstream switch or container recreate.
    Wait until /health returns 200, or fail after timeout.
    """
    deadline = time.time() + timeout_s
    last: Optional[CurlResult] = None

    while time.time() < deadline:
        res = curl_health()
        last = res
        if res.status_code == 200:
            return
        time.sleep(sleep_s)

    last = last or CurlResult(status_code=0, body="")
    raise AssertionError(
        f"Health never became 200 within {timeout_s}s. Last: {last.status_code} body={last.body}"
    )


def hit(n: int) -> None:
    run(
        f'for i in $(seq 1 {n}); do '
        f'curl -s --connect-timeout 1 --max-time 2 "{BASE_URL}{HEALTH_PATH}" >/dev/null; '
        f"done",
        check=False,
    )


def smoke_ok() -> None:
    # Robust against transient 502 right after upstream switch/reload.
    wait_for_health_ok(timeout_s=30.0, sleep_s=0.25)


# -----------------------------
# Stack lifecycle
# -----------------------------
def recreate_green() -> None:
    compose(f"build {GREEN_SERVICE}", check=False)
    compose(f"up -d --no-deps --force-recreate {GREEN_SERVICE}")


# -----------------------------
# Nginx log helpers (deterministic)
# -----------------------------
def nginx_log_lines(tail: int = 5000) -> List[str]:
    logs = compose(f"logs --tail={tail} {NGINX_SERVICE}", check=False)
    return logs.splitlines()


def label_upstreams(lines: List[str], blue_ip: str, green_ip: str) -> List[str]:
    out: List[str] = []
    for ln in lines:
        ln = ln.replace(f"upstream={blue_ip}:8000", f"upstream={BLUE_SERVICE}:8000")
        ln = ln.replace(f"upstream={green_ip}:8000", f"upstream={GREEN_SERVICE}:8000")
        out.append(ln)
    return out


def new_health_lines_since(baseline_len: int, blue_ip: str, green_ip: str) -> List[str]:
    after = nginx_log_lines()
    new_lines = after[baseline_len:] if baseline_len <= len(after) else after
    new_lines = label_upstreams(new_lines, blue_ip=blue_ip, green_ip=green_ip)
    return [ln for ln in new_lines if "GET /health" in ln and " upstream=" in ln]


def count_health_upstreams(lines: List[str]) -> Dict[str, int]:
    counts = {BLUE_SERVICE: 0, GREEN_SERVICE: 0}
    for ln in lines:
        if f"upstream={BLUE_SERVICE}:8000" in ln:
            counts[BLUE_SERVICE] += 1
        if f"upstream={GREEN_SERVICE}:8000" in ln:
            counts[GREEN_SERVICE] += 1
    return counts


def assert_most_traffic(expected_service: str, phase_lines: List[str], min_ratio: float) -> None:
    counts = count_health_upstreams(phase_lines)
    total = counts[BLUE_SERVICE] + counts[GREEN_SERVICE]
    expected = counts[expected_service]

    print("\n--- NGINX new /health lines (this phase) ---")
    for ln in phase_lines[-min(40, len(phase_lines)) :]:
        print(ln)

    if total == 0:
        raise AssertionError("No new /health log lines observed for this phase.")

    ratio = expected / total
    if ratio < min_ratio:
        raise AssertionError(
            f"Routing too mixed: expected mostly {expected_service} but got counts={counts}, "
            f"ratio={ratio:.2f} < {min_ratio:.2f}"
        )

    print(f"\nUpstream counts OK (this phase): {counts} (expected mostly {expected_service})")


def warmup_and_baseline() -> int:
    """
    Make a single request to ensure NGINX workers are using the new upstream,
    then set the baseline AFTER that warmup so our counts only include the hit() calls.
    """
    baseline = len(nginx_log_lines())
    run(f'curl -s --connect-timeout 1 --max-time 2 "{BASE_URL}{HEALTH_PATH}" >/dev/null', check=False)
    return len(nginx_log_lines())


# -----------------------------
# The test
# -----------------------------
def test_blue_green_deployment_e2e():
    # Bring stack up
    compose("up -d --build --remove-orphans")

    blue_ip = get_ip(BLUE_SERVICE)
    green_ip = get_ip(GREEN_SERVICE)
    print(f"\nIP mapping: {BLUE_SERVICE}={blue_ip} | {GREEN_SERVICE}={green_ip}")

    # -------------------------
    # Phase 1: Route to BLUE
    # -------------------------
    set_upstream(BLUE_SERVICE)
    smoke_ok()

    baseline = warmup_and_baseline()
    hit(N_HITS)

    phase1 = new_health_lines_since(baseline, blue_ip=blue_ip, green_ip=green_ip)
    assert_most_traffic(BLUE_SERVICE, phase1, MIN_RATIO)

    print("\n--- compose ps before green recreate ---")
    print(compose("ps", check=False))

    # -------------------------
    # Phase 2: Recreate GREEN only
    # -------------------------
    recreate_green()

    # Wait for the stack to stabilize (green recreate can briefly break upstream health)
    smoke_ok()

    green_ip = get_ip(GREEN_SERVICE)

    print("\n--- compose ps after green recreate ---")
    print(compose("ps", check=False))

    # -------------------------
    # Phase 3: Route to GREEN
    # -------------------------
    set_upstream(GREEN_SERVICE)
    smoke_ok()

    baseline = warmup_and_baseline()
    hit(N_HITS)

    phase3 = new_health_lines_since(baseline, blue_ip=blue_ip, green_ip=green_ip)
    assert_most_traffic(GREEN_SERVICE, phase3, MIN_RATIO)