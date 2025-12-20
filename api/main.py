from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from inference.predictor import predict
from src.website_feature_extraction import FeatureExtractor

app = FastAPI(title="PhishGuard AI", version="1.0")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "api" / "templates"))

extractor = FeatureExtractor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "url": "",
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    url: str = Form(...),
    model_type: str = Form("xgboost"),
):
    try:
        # URL → Feature Extraction
        features = extractor.extract(url)

        # ML Prediction
        result = predict(features, model_type=model_type)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "url": url,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {
                    "model": model_type,
                    "prediction": 0,
                    "result_text": f"Error: {str(e)}",
                    "phishing_probability": None,
                },
                "url": url,
            },
        )
