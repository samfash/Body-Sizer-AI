from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


REQUEST_COUNT = Counter("ai_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("ai_request_latency_seconds", "Latency", ["endpoint"])


def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
