from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# IMPORTANT: predictor
from inference.predictor import predict

app = FastAPI(title="PhishGuard AI", version="1.0")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "api" / "templates"))

# ===============================
# UI → MODEL VALUE MAPPING
# ===============================
UI_TO_MODEL_MAP = {
    "yes": 1,
    "no": 0,
    "unknown": -1
}

# ===============================
# FEATURE LIST (ORDER MATTERS)
# ===============================
FEATURES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",  # dataset column name
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]


# ===============================
# HOME PAGE
# ===============================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    default_values = {f: "unknown" for f in FEATURES}
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURES,
            "values": default_values,
            "result": None,
            "errors": [],
            "model_type": "xgboost",
        },
    )


# ===============================
# PREDICTION (UI FORM)
# ===============================
@app.post("/predict", response_class=HTMLResponse)
async def predict_ui(request: Request, model_type: str = Form("xgboost")):
    form = await request.form()

    input_data = {}
    errors = []

    for feature in FEATURES:
        raw_value = form.get(feature, "unknown").lower()

        if raw_value not in UI_TO_MODEL_MAP:
            errors.append(f"Invalid value for {feature}")
        else:
            input_data[feature] = UI_TO_MODEL_MAP[raw_value]

    result = None
    if not errors:
        try:
            result = predict(input_data, model_type=model_type)
        except Exception as e:
            errors.append(str(e))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": FEATURES,
            "values": {f: form.get(f, "unknown") for f in FEATURES},
            "result": result,
            "errors": errors,
            "model_type": model_type,
        },
    )


# ===============================
# JSON API (POSTMAN / FRONTEND)
# ===============================
@app.post("/predict_json")
async def predict_json(payload: dict):
    model_type = payload.get("model_type", "xgboost")
    ui_features = payload.get("features", {})

    input_data = {}
    for k, v in ui_features.items():
        input_data[k] = UI_TO_MODEL_MAP.get(v.lower(), -1)

    return predict(input_data, model_type=model_type)
