import logging

from pmhras.data.audit import run_audit ,load_dataset
from pmhras.preprocessing.pipeline import preprocessing
from pmhras.preprocessing.split import data_split
from pmhras.models.train import train_model
from pmhras.models.evaluate import evaluate_predictions
from pmhras.config.settings import load_config

def run_project():
    config = load_config("configs/data_audit.yaml")

    audit_results = run_audit(config)

    df = load_dataset(config)

    x_train , x_test , y_train , y_test = data_split(df)

    feature_config = config['preprocessing'] 

    preprocessor = preprocessing(
                            feature_config['numerical'].get("numerical_columns"),
                            feature_config['ordinal'].get("ordinal_columns"),
                            feature_config['nominal'].get("nominal_columns"),
                            feature_config['ordinal'].get("ordinal_categorie"),
                            )

    fitted_models = train_model(preprocessor,x_train,y_train)

    result = evaluate_predictions(fitted_models,x_test,y_test)

    