"""Conservative, stateless extraction from observed service message formats."""

import ipaddress
import re
from collections import Counter, defaultdict
from urllib.parse import parse_qs

from .ingest import Enrichment, NormalizedLog, _integer

KV = re.compile(r'(?:^|\s)([a-z_]+)=("[^"\n]*"|[^\s]+)')
LOGIN = re.compile(r"^Login (succeeded|failed) for user ([^\s:()]+)(?=[:\s]|$)")
AUTH_USER = re.compile(
    r"^(?:Access token refreshed for user |Session \S+ closed for user |"
    r"Refresh token expired or revoked for user |Logout rejected for user |"
    r"Token rejected for user |Clock skew detected for token of user )([^\s:()]+)"
)
HEALTH = re.compile(r"^(GET|HEAD) /health -> [0-9]{3}(?:\s|$)")
METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}


def message_template(message: str) -> str | None:
    """Replace syntactic variables, without learning vocabulary from the file."""
    if not message.strip():
        return None

    def address(match: re.Match) -> str:
        candidate = match.group(0)
        try:
            ipaddress.ip_address(candidate)
            return "<IP>"
        except ValueError:
            return candidate

    result = re.sub(r"(?<![\w:])[0-9a-fA-F:.]*[.:][0-9a-fA-F:.]+(?![\w:])", address, message)
    result = re.sub(r"\b(user\s+)[^\s:()]+", r"\1<ID>", result)
    result = re.sub(r"\b(req_id=|request_id=|session=|txn=|sku=)[^\s]+", r"\1<ID>", result)
    result = re.sub(r"\b(?:ORD|SKU|TXN)-[\w-]+\b|\b[0-9a-fA-F]{8,}\b|\bu[0-9]+\b", "<ID>", result)
    return re.sub(r"[0-9]+(?:\.[0-9]+)?", "<NUM>", result)


def _gateway(message: str, fields: Enrichment) -> str:
    values = dict(KV.findall(message))
    method = values.get("method")
    if method in METHODS:
        fields.method = method
    path = values.get("path")
    if path and path.startswith("/"):
        fields.message_path, separator, query = path.partition("?")
        fields.message_query = query if separator else None
        fields.query_parameters = parse_qs(query, keep_blank_values=True) if separator else None
    fields.upstream_service = values.get("upstream")
    fields.request_id = values.get("req_id")
    fields.upstream_latency_ms = _integer(values.get("upstream_latency_ms", ""), 0)
    recognized = {"method", "path", "upstream", "req_id", "upstream_latency_ms"} & values.keys()
    if not recognized:
        return "unmatched"
    valid = fields.method is not None and fields.message_path is not None
    if "upstream_latency_ms" in values and fields.upstream_latency_ms is None:
        valid = False
    return "matched" if valid else "partial"


def _search(message: str, fields: Enrichment) -> str:
    values = dict(KV.findall(message))
    # Anchor context so arbitrary text mentioning a query or a duration is not
    # silently interpreted as a search operation.
    recognized = message.startswith(('query="', 'suggest prefix="', 'product lookup sku=', 'product sku='))
    if not recognized:
        return "unmatched"
    metadata = {}
    for key in ("query", "prefix"):
        value = values.get(key)
        if value and value.startswith('"') and value.endswith('"'):
            if key == "query":
                fields.query = value[1:-1]
            else:
                metadata[key] = value[1:-1]
    if "sku" in values:
        metadata["sku"] = values["sku"]
    for key in ("hits", "results", "shards"):
        if key in values:
            number = _integer(values[key], 0)
            if number is not None:
                metadata[key] = number
    took = values.get("took", "")
    if took.endswith("ms"):
        fields.took_ms = _integer(took[:-2], 0)
    fields.query_metadata = metadata or None
    identity = fields.query is not None or "prefix" in metadata or "sku" in metadata
    optional_valid = all(key not in values or key in metadata for key in ("hits", "results", "shards"))
    if "took" in values and fields.took_ms is None:
        optional_valid = False
    # Recognized cached/missing-product messages legitimately have no duration.
    complete = fields.took_ms is not None or "not modified (etag match)" in message or message.endswith(" not found")
    return "matched" if identity and optional_valid and complete else "partial"


def _auth(message: str, fields: Enrichment) -> str:
    login = LOGIN.match(message)
    if login:
        fields.user_id = login.group(2)
        fields.authentication_outcome = "success" if login.group(1) == "succeeded" else "failure"
        return "matched"
    user = AUTH_USER.match(message)
    if user:
        fields.user_id = user.group(1)
        # Refresh/logout/token problems are not failed password attempts.
        return "matched"
    if message.startswith(("Login ", "Rejected login request:", "Unhandled exception in LoginHandler:",
                           "Unexpected error validating credentials:")):
        return "partial"
    return "unmatched"


def parse_messages(records: list[NormalizedLog]) -> dict[str, dict]:
    """Enrich in place, preserving raw text and identifiers; return service coverage."""
    counters: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        fields = Enrichment(message_template=message_template(record.message))
        if not record.message.strip():
            status = "empty"
        elif HEALTH.match(record.message):
            status = "matched"
        elif record.service == "api-gateway":
            status = _gateway(record.message, fields)
        elif record.service == "search-service":
            status = _search(record.message, fields)
        elif record.service == "auth-service":
            status = _auth(record.message, fields)
        else:
            status = "unmatched"
        record.enrichment = fields
        record.parse_status = status
        counters[record.service][status] += 1
    coverage = {}
    for service, counts in sorted(counters.items()):
        total = sum(counts.values())
        coverage[service] = {
            "total": total,
            **{status: counts[status] for status in ("matched", "partial", "unmatched", "empty")},
            "matched_fraction": counts["matched"] / total,
        }
    return coverage
