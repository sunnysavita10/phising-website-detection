from src.config_loader import load_config
from src.data_loader import load_dataset
from src.preprocessor import preprocess_data
from src.train_xgboost import train_xgboost
from src.train_ann import train_ann

def run_pipeline():
    config = load_config()

    df = load_dataset(config["data"]["file_path"])

    X_train, X_test, y_train, y_test = preprocess_data(
        df=df,
        target_col=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        scaler_path=f"{config['artifacts']['directory']}/{config['artifacts']['scaler_filename']}"
    )

    train_xgboost(
        X_train, X_test, y_train, y_test,
        params=config["xgboost"],
        save_path=f"{config['artifacts']['directory']}/{config['artifacts']['xgb_model_filename']}"
    )

    train_ann(
        X_train, X_test, y_train, y_test,
        params=config["mlp"],
        save_path=f"{config['artifacts']['directory']}/{config['artifacts']['ann_model_filename']}"
    )

    print("\n Pipeline Completed Successfully!")
