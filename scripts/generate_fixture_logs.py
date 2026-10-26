"""Seeded, labelled synthetic e-commerce logs for evaluation (Phase 6).

Writes a log CSV with exactly the eight source columns and, separately, an
evaluator-only labels JSON. Detectors receive only the CSV; the detector package
never imports this module. Event times, services, IPs, and intensities are drawn
from the seed, so no detector can depend on specific injected identities.

    python scripts/generate_fixture_logs.py --seed 11 --output artifacts/fixtures/seed-11
"""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "1.0"
COLUMNS = ("timestamp", "service", "log_level", "http_status", "response_time_ms",
           "endpoint", "source_ip", "message")
FIRST_MONDAY = datetime(2026, 11, 2)
# Relative hourly volume (weekday shape of the supplied week); weekends are lower.
HOURLY = (255, 286, 265, 253, 278, 305, 347, 435, 598, 743, 880, 1035,
          1171, 1203, 1230, 1063, 881, 784, 697, 778, 683, 553, 395, 298)
WEEKEND_FACTOR = 0.72
# Business records per weekday (health checks excluded), close to the supplied file.
DAILY = {"api-gateway": 3800, "auth-service": 2000, "orders-service": 2575,
         "payment-service": 1740, "search-service": 2730}
# endpoint, weight, median latency ms, upstream (gateway only)
ENDPOINTS = {
    "api-gateway": (("/api/search", 0.34, 180, "search-service"), ("/api/orders", 0.23, 160, "orders-service"),
                    ("/api/users/profile", 0.18, 80, "auth-service"),
                    ("/api/payments", 0.11, 600, "payment-service"),
                    ("/api/products", 0.14, 150, "search-service")),
    "auth-service": (("/login", 0.46, 180, None), ("/api/auth/token/refresh", 0.40, 35, None),
                     ("/logout", 0.14, 20, None)),
    "orders-service": (("/api/orders", 0.57, 120, None), ("/api/cart", 0.29, 70, None),
                       ("/api/orders/status", 0.14, 40, None)),
    "payment-service": (("/api/payments", 0.57, 550, None), ("/api/payments/status", 0.35, 50, None),
                        ("/api/payments/refund", 0.08, 400, None)),
    "search-service": (("/api/search", 0.59, 150, None), ("/api/search/suggest", 0.25, 90, None),
                       ("/api/products", 0.16, 120, None)),
}
HEALTH = {"api-gateway": "method=GET path=/health status={status} upstreams_healthy=4/4",
          "auth-service": "GET /health -> {status} (redis=ok, user-db=ok)",
          "orders-service": "GET /health -> {status} (db=ok, inventory=ok)",
          "payment-service": "GET /health -> {status} (db=ok, provider=ok)",
          "search-service": "GET /health -> {status} (cluster=green, shards=6/6)"}
QUERIES = ("laptop stand", "usb c cable", "gaming mouse", "sunglasses", "screen protector", "desk lamp")
DOWNSTREAM = ("orders-service", "payment-service", "search-service")


class Generator:
    def __init__(self, seed: int, scale: float):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.start = FIRST_MONDAY + timedelta(days=7 * (seed % 5))
        octets = self.rng.integers(1, 255, size=(3000, 3))
        self.clients = [f"10.{a}.{b}.{c}" for a, b, c in octets[:2000]] + [
            f"192.168.{a}.{c}" for a, _, c in octets[2000:2700]] + [f"203.0.113.{c}" for _, _, c in octets[2700:]]
        self.rows: list[dict] = []

    def hex(self, n: int) -> str:
        return f"{int(self.rng.integers(0, 16 ** n)):0{n}x}"

    def user(self) -> str:
        return f"u{self.rng.integers(10000, 99999)}"

    def latency(self, median: float) -> int:
        value = median * self.rng.lognormal(0, 0.35)
        if self.rng.random() < 0.005:  # occasional slow request: normal background
            value *= self.rng.uniform(4, 10)
        return max(1, int(value))

    def message(self, service: str, endpoint: str, status: int, latency: int, upstream: str | None,
                method: str) -> tuple[str, str]:
        """Return (log_level, message) using the supplied services' observed formats."""
        level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
        if service == "api-gateway":
            path = endpoint + (f"?q={self.rng.choice(QUERIES).replace(' ', '+')}" if endpoint == "/api/search" else "")
            return level, (f"method={method} path={path} upstream={upstream} status={status} "
                           f"upstream_latency_ms={max(1, latency - 3)} req_id={self.hex(12)}")
        if service == "auth-service":
            if endpoint == "/login":
                return (("INFO", f"Login succeeded for user {self.user()} (auth=password, mfa=false) "
                                 f"session={self.hex(8)}") if status < 400 else
                        ("WARN", f"Login failed for user {self.user()}: invalid credentials") if status < 500 else
                        (level, "Unexpected error validating credentials: user-db timeout"))
            if endpoint == "/logout":
                return level, f"Session {self.hex(8)} closed for user {self.user()}"
            return level, f"Access token refreshed for user {self.user()} expires_in=900s"
        if status >= 500:
            return level, {"orders-service": "OrderService - failed to create order: inventory timeout",
                           "payment-service": f"txn=TXN-{self.hex(10)} provider timeout; payment failed",
                           "search-service": "search backend error: shard failure"}[service]
        if service == "orders-service":
            if endpoint == "/api/cart":
                return level, f"CartService - cart updated for user {self.user()} sku=SKU-{self.rng.integers(10000, 99999)} qty=2"
            if endpoint == "/api/orders/status":
                return level, f"OrderQueryService - order ORD-{self.rng.integers(10**6, 10**7)} status=shipped"
            return level, (f"OrderService - order ORD-{self.rng.integers(10**6, 10**7)} created for user "
                           f"{self.user()} items=2 total={self.rng.uniform(5, 300):.2f} USD")
        if service == "payment-service":
            txn = f"txn=TXN-{self.hex(10)}"
            if endpoint == "/api/payments/status":
                return level, f"{txn} lookup status=authorized"
            if endpoint == "/api/payments/refund":
                return level, f"{txn} refund issued amount={self.rng.uniform(5, 100):.2f}"
            return level, f"{txn} status=authorized amount={self.rng.uniform(5, 400):.2f} currency=EUR provider=acq-gw-1"
        if status == 404:
            return level, f"product sku=SKU-{self.rng.integers(10000, 99999)} not found"
        if endpoint == "/api/search/suggest":
            return level, f'suggest prefix="lap" results={self.rng.integers(1, 10)} took={latency}ms'
        if endpoint == "/api/products":
            return level, f"product lookup sku=SKU-{self.rng.integers(10000, 99999)} found took={latency}ms"
        return level, f'query="{self.rng.choice(QUERIES)}" hits={self.rng.integers(0, 900)} took={latency}ms shards=6'

    def add(self, when: datetime, service: str, endpoint: str, ip: str, status: int | None = None,
            latency: int | None = None) -> dict:
        spec = next(e for e in ENDPOINTS[service] if e[0] == endpoint)
        latency = self.latency(spec[2]) if latency is None else latency
        if status is None:
            draw = self.rng.random()
            failure = 0.07 if endpoint == "/login" else 0.05 if endpoint == "/api/products" else 0.015
            status = 500 if draw < 0.003 else (401 if endpoint == "/login" else 404) if draw < 0.003 + failure else (
                201 if endpoint in ("/api/payments",) else 204 if endpoint == "/logout" else 200)
        method = "POST" if endpoint in ("/api/payments", "/login") else "GET"
        level, message = self.message(service, endpoint, status, latency, spec[3], method)
        row = {"timestamp": when, "service": service, "log_level": level, "http_status": status,
               "response_time_ms": latency, "endpoint": endpoint, "source_ip": ip, "message": message,
               "upstream": spec[3]}
        self.rows.append(row)
        return row

    def background(self, days: int) -> None:
        total = sum(HOURLY)
        for minute in range(days * 24 * 60):
            when = self.start + timedelta(minutes=minute)
            factor = WEEKEND_FACTOR if when.weekday() >= 5 else 1.0
            for service, daily in DAILY.items():
                rate = daily * self.scale * HOURLY[when.hour] / total / 60 * factor
                specs = ENDPOINTS[service]
                weights = [s[1] / sum(e[1] for e in specs) for s in specs]
                for _ in range(self.rng.poisson(rate)):
                    endpoint = specs[int(self.rng.choice(len(specs), p=weights))][0]
                    self.add(when + timedelta(seconds=int(self.rng.integers(0, 60))), service, endpoint,
                             self.clients[int(self.rng.integers(len(self.clients)))])
                if minute % 5 == 0:  # load-balancer health checks
                    lb = "10.0.0.2" if (minute // 5) % 2 else "10.0.0.3"
                    self.rows.append({"timestamp": when + timedelta(seconds=int(self.rng.integers(0, 30))),
                                      "service": service, "log_level": "INFO", "http_status": 200,
                                      "response_time_ms": int(self.rng.integers(2, 9)), "endpoint": "/health",
                                      "source_ip": lb, "message": HEALTH[service].format(status=200),
                                      "upstream": None})

    def affected(self, service: str, start: datetime, end: datetime, gateway: bool) -> list[dict]:
        return [r for r in self.rows if start <= r["timestamp"] < end and r["endpoint"] != "/health"
                and (r["service"] == service or (gateway and r["service"] == "api-gateway"
                                                 and r["upstream"] == service))]

    def rewrite(self, row: dict, status: int | None = None, latency: int | None = None,
                level: str | None = None, message: str | None = None) -> None:
        if status is not None:
            row["http_status"] = status
        if latency is not None:
            row["response_time_ms"] = latency
        method = "POST" if row["endpoint"] in ("/api/payments", "/login") else "GET"
        row["log_level"], row["message"] = self.message(row["service"], row["endpoint"], row["http_status"],
                                                        row["response_time_ms"], row["upstream"], method)
        if level is not None:
            row["log_level"], row["message"] = level, message


def _minutes(rng, low: int, high: int) -> timedelta:
    return timedelta(minutes=int(rng.integers(low, high + 1)), seconds=int(rng.integers(0, 60)))


def generate(seed: int, days: int = 7, scale: float = 1.0) -> tuple[list[dict], dict]:
    """Return (csv rows, labels). `scale` multiplies background volume (tests only)."""
    if days < 3:
        raise ValueError("fixtures need at least training, validation, and test days")
    g = Generator(seed, scale)
    rng = g.rng
    g.background(days)
    events, labels = [], []
    validation_day, test_day = days - 3, days - 2  # chronological 4/1/2 for seven days
    # Busy-hour two-hour slots per day keep events on one day from overlapping.
    free = {day: [9, 11, 13, 15, 17, 19] for day in range(days)}

    def slot(day: int) -> datetime:
        hours = free[day]
        hour = hours.pop(int(rng.integers(len(hours))))
        return g.start + timedelta(days=day, hours=hour) + _minutes(rng, 0, 50)

    plan = [(day, str(rng.choice(["http_errors", "latency", "service_errors"]))) for day in range(validation_day)]
    plan += [(validation_day, "http_errors"), (validation_day, "latency")]
    test_categories = ["http_errors", "latency", "service_errors", "traffic"]
    rng.shuffle(test_categories)
    plan += [(test_day + i % 2, c) for i, c in enumerate(test_categories)]
    batch_service = str(rng.choice(["orders-service", "payment-service"]))
    for day, category in plan:
        start = slot(day)
        service = str(rng.choice(DOWNSTREAM))
        if category == "http_errors":
            end = start + _minutes(rng, 15, 40)
            fraction = float(rng.uniform(0.3, 0.7))
            for row in g.affected(service, start, end, gateway=True):
                if rng.random() < fraction:
                    g.rewrite(row, status=502 if row["service"] == "api-gateway" else int(rng.choice([500, 503])))
            events.append((category, service, start, end, {"fraction_5xx": round(fraction, 3)}, True))
        elif category == "latency":
            end = start + _minutes(rng, 20, 45)
            factor = float(rng.uniform(4, 8))
            for row in g.affected(service, start, end, gateway=True):
                g.rewrite(row, latency=int(row["response_time_ms"] * factor))
            events.append((category, service, start, end, {"latency_factor": round(factor, 2)}, True))
        elif category == "service_errors":
            end = start + _minutes(rng, 20, 40)
            fraction = float(rng.uniform(0.25, 0.5))
            text = {"orders-service": "OrderService - inventory sync failed; served cached stock level",
                    "payment-service": "PaymentService - ledger write failed; queued for retry",
                    "search-service": "SearchService - index refresh failed; serving stale results"}[service]
            for row in g.affected(service, start, end, gateway=False):
                if rng.random() < fraction:
                    g.rewrite(row, level="ERROR", message=text)
            events.append((category, service, start, end, {"error_fraction": round(fraction, 3)}, False))
        else:  # non-recurring scraper surge through the gateway to search
            start = g.start + timedelta(days=day, hours=int(rng.integers(4, 7))) + _minutes(rng, 0, 50)
            end = start + _minutes(rng, 30, 75)
            ip, per_minute = f"198.51.100.{rng.integers(2, 250)}", float(rng.uniform(8, 14))
            when = start
            while when < end:
                for _ in range(rng.poisson(per_minute)):
                    at = when + timedelta(seconds=int(rng.integers(0, 60)))
                    if at < end:
                        latency = g.latency(150)
                        g.add(at, "api-gateway", "/api/products", ip, 200, latency)
                        g.add(at, "search-service", "/api/products", ip, 200, max(1, latency - 5))
                when += timedelta(minutes=1)
            events.append((category, "search-service", start, end,
                           {"source_ip": ip, "requests_per_minute": round(per_minute, 2)}, True))
    # Difficult low-volume case: a night-time 5xx burst on a sparse service.
    weak_service = str(rng.choice([s for s in DOWNSTREAM if s != batch_service]))
    start = g.start + timedelta(days=test_day + 1, hours=int(rng.integers(1, 4))) + _minutes(rng, 0, 50)
    end = start + _minutes(rng, 20, 30)
    for row in g.affected(weak_service, start, end, gateway=False):
        if rng.random() < 0.6:
            g.rewrite(row, status=500)
    events.append(("http_errors", weak_service, start, end, {"fraction_5xx": 0.6, "difficulty": "weak"}, False))
    # Repeated failed logins from one IP, starting just before the validation/test boundary.
    start = g.start + timedelta(days=test_day) - _minutes(rng, 5, 15)
    end = start + _minutes(rng, 25, 45)
    ip, per_minute = f"203.0.113.{rng.integers(2, 250)}", float(rng.uniform(3, 8))
    when = start
    while when < end:
        for _ in range(rng.poisson(per_minute)):
            at = when + timedelta(seconds=int(rng.integers(0, 60)))
            if at < end:
                g.add(at, "auth-service", "/login", ip, 401)
        when += timedelta(minutes=1)
    labels.append({"category": "auth_failures", "service": "auth-service", "endpoint": None, "source_ip": ip,
                   "start": start, "end": end, "difficulty": "standard",
                   "injection": {"failures_per_minute": round(per_minute, 2)}})
    for category, service, start, end, injection, gateway in events:
        difficulty = injection.pop("difficulty", "standard")
        labels.append({"category": category, "service": service, "endpoint": None, "source_ip": None,
                       "start": start, "end": end, "difficulty": difficulty, "injection": injection})
        if gateway:  # the gateway logs the same requests; contracts keep it as its own linked incident
            labels.append({"category": category, "service": "api-gateway", "endpoint": None, "source_ip": None,
                           "start": start, "end": end, "difficulty": difficulty, "injection": injection})
    # Normal recurring behaviour (unlabelled): a nightly export batch at a seed-chosen clock time.
    clock = timedelta(hours=int(rng.integers(0, 5)), minutes=int(rng.integers(0, 12)) * 5)
    batch_size = int(rng.integers(40, 60))
    for day in range(days):
        for page in range(batch_size + int(rng.integers(-5, 6))):
            at = g.start + timedelta(days=day) + clock + timedelta(seconds=int(rng.integers(0, 240)))
            row = g.add(at, batch_service, next(e[0] for e in ENDPOINTS[batch_service]
                                                if e[0] in ("/api/orders", "/api/payments/status")), "10.20.4.15")
            row.update(http_status=200, log_level="INFO",
                       message=f"export batch page={page + 1} records=500 job=nightly-export")
    labels.sort(key=lambda e: (e["start"], e["service"], e["category"]))
    for index, label in enumerate(labels, 1):
        label["event_id"] = f"s{seed}-e{index:02d}"
        label["start"], label["end"] = label["start"].isoformat(), label["end"].isoformat()
    g.rows.sort(key=lambda r: r["timestamp"])  # stable: equal timestamps keep generation order
    rows = [{**{c: r[c] for c in COLUMNS}, "timestamp": r["timestamp"].isoformat(sep=" ")} for r in g.rows]
    return rows, {
        "schema_version": "1.0", "generator": "scripts/generate_fixture_logs.py",
        "generator_version": GENERATOR_VERSION, "seed": seed, "start_day": g.start.date().isoformat(),
        "days": days, "scale": scale, "events": labels,
        "normal_patterns": [{"kind": "recurring_batch", "service": batch_service, "source_ip": "10.20.4.15",
                             "daily_clock_time": str(clock), "records_per_run": batch_size}],
    }


def write(seed: int, output: Path, days: int = 7, scale: float = 1.0) -> tuple[Path, Path]:
    """Write `logs.csv` and the evaluator-only `labels.json` into a new directory."""
    rows, labels = generate(seed, days, scale)
    output.mkdir(parents=True)
    logs, label_path = output / "logs.csv", output / "labels.json"
    with logs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    label_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return logs, label_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    print(*write(args.seed, args.output, args.days), sep="\n")
