from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictionResponse, HealthResponse
from api.inference import predict_from_image_bytes
import uvicorn, os
from observability.metrics import REQUEST_COUNT, REQUEST_LATENCY, metrics_endpoint
from observability.tracing import init_tracer
from observability.logger import logger

app = FastAPI(title="Body Measurement Inference API")
init_tracer("ai-service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route("/metrics", metrics_endpoint)

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(image: UploadFile = File(...)):
    contents = await image.read()

    with REQUEST_LATENCY.labels(endpoint="/predict-image").time():
        try:
            res = predict_from_image_bytes(contents)
            REQUEST_COUNT.labels(endpoint="/predict-image", status="200").inc()
            return JSONResponse({"success": True, "predictions": res})
        except Exception as e:
            logger.exception("inference failure")
            REQUEST_COUNT.labels(endpoint="/predict-image", status="500").inc()
            raise HTTPException(status_code=400, detail=str(e))
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
