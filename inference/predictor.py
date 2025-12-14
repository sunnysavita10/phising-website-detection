import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.config_loader import load_config


# =========================================================
# PATHS + LOAD ARTIFACTS (config-driven)
# =========================================================

CONFIG = load_config()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / CONFIG["artifacts"]["directory"]

SCALER_PATH = ARTIFACTS_DIR / CONFIG["artifacts"]["scaler_filename"]
XGB_MODEL_PATH = ARTIFACTS_DIR / CONFIG["artifacts"]["xgb_model_filename"]
ANN_MODEL_PATH = ARTIFACTS_DIR / CONFIG["artifacts"]["ann_model_filename"]


def _safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Fix: Run training pipeline first so artifacts get created."
        )
    return joblib.load(path)


SCALER = _safe_load(SCALER_PATH)
XGB_MODEL = _safe_load(XGB_MODEL_PATH)
ANN_MODEL = _safe_load(ANN_MODEL_PATH)


# =========================================================
# HELPERS
# =========================================================

def decode_label(pred: int) -> str:
    return "Legit Website" if int(pred) == 1 else "Phishing Website"


def _get_expected_columns_from_scaler() -> Optional[list]:
    """
    Try best to recover expected feature columns.
    Best case: scaler was fit on a DataFrame and retains feature names.
    Otherwise returns None.
    """
    if hasattr(SCALER, "feature_names_in_"):
        return list(SCALER.feature_names_in_)
    return None


EXPECTED_COLUMNS = _get_expected_columns_from_scaler()


def validate_and_build_df(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert dict -> DataFrame and ensure it matches training feature columns.

    - If scaler has feature_names_in_: reorder & validate strictly
    - Else: just create DataFrame from dict (assumes correct order/keys provided)
    """
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a Python dict of feature_name -> value.")

    df = pd.DataFrame([input_dict])

    if EXPECTED_COLUMNS is not None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}\n"
                f"Expected columns (total {len(EXPECTED_COLUMNS)}): {EXPECTED_COLUMNS}"
            )

        # We allow extra columns but drop them to be safe
        if extra:
            df = df.drop(columns=extra)

        # Reorder exactly as training
        df = df[EXPECTED_COLUMNS]

    return df


def preprocess_input(input_dict: Dict[str, Any]) -> np.ndarray:
    """
    input_dict -> DataFrame -> scaled numpy array
    """
    df = validate_and_build_df(input_dict)
    X_scaled = SCALER.transform(df)
    return X_scaled


def predict(
    input_dict: Dict[str, Any],
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Predict phishing vs legit.

    model_type: "xgboost" | "ann"
    returns: dict with prediction + label + probability (risk score)
    """
    X_scaled = preprocess_input(input_dict)

    if model_type.lower() == "xgboost":
        model = XGB_MODEL
    elif model_type.lower() in ["ann", "mlp", "mlpclassifier"]:
        model = ANN_MODEL
        model_type = "ann"
    else:
        raise ValueError("model_type must be 'xgboost' or 'ann'.")

    pred = int(model.predict(X_scaled)[0])

    # Risk score: probability of class 1 (Legit). We'll also provide phishing_score.
    legit_prob = None
    phishing_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]  # [prob_class0, prob_class1]
        phishing_prob = float(proba[0])  # class 0 = phishing
        legit_prob = float(proba[1])     # class 1 = legit

    return {
        "model": model_type,
        "prediction": pred,                      # 1 legit, 0 phishing
        "result_text": decode_label(pred),
        "phishing_probability": phishing_prob,   # higher => more risky
        "legit_probability": legit_prob
    }


# =========================================================
# CLI DEMO
# =========================================================

if __name__ == "__main__":
    # NOTE:
    # Use same feature names as training CSV (excluding target column).
    # Values should be numeric (mostly -1/0/1 for this dataset).

    # sample_input = {
    #     "having_IP_Address": 0,
    #     "URL_Length": 1,
    #     "Shortining_Service": 0,
    #     "having_At_Symbol": 0,
    #     "double_slash_redirecting": 1,
    #     "Prefix_Suffix": 0,
    #     "having_Sub_Domain": 1,
    #     "SSLfinal_State": 1,
    #     "Domain_registeration_length": 1,
    #     "Favicon": 1,
    #     "port": 0,
    #     "HTTPS_token": 0,
    #     "Request_URL": 1,
    #     "URL_of_Anchor": 1,
    #     "Links_in_tags": 1,
    #     "SFH": 1,
    #     "Submitting_to_email": 0,
    #     "Abnormal_URL": 0,
    #     "Redirect": 1,
    #     "on_mouseover": 0,
    #     "RightClick": 1,
    #     "popUpWidnow": 0,
    #     "Iframe": 0,
    #     "age_of_domain": 1,
    #     "DNSRecord": 1,
    #     "web_traffic": 1,
    #     "Page_Rank": 1,
    #     "Google_Index": 1,
    #     "Links_pointing_to_page": 1,
    #     "Statistical_report": 0
    # }
    from feature_extraction import FeatureExtractor

    extractor = FeatureExtractor()
    #url = "https://secure-paypal-login-verification.xyz/login"
    #url = "http://google.com.secure-login.verify-account.example.com/"
    url = "http://google.com/"
    features = extractor.extract(url)
    # print("\nExtracted Features:")
    # for k, v in features.items():
    #     print(f"  {k}: {v}")

    print("\n--- XGBoost Prediction ---")
    print(predict(features, model_type="xgboost"))

    print("\n--- ANN (MLPClassifier) Prediction ---")
    print(predict(features, model_type="ann"))
