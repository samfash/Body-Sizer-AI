from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
REQUEST_COUNT = Counter('ai_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('ai_request_latency_seconds', 'Latency', ['endpoint'])

from fastapi import Response

def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
