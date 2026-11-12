import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.base import clone
from pmhras.models.exceptions import ModelTrainingError
from sklearn.compose import ColumnTransformer
from pmhras.config.settings import load_config

logger = logging.getLogger(__name__)

def classification_models(preprocessor: ColumnTransformer):

    config = load_config("configs/data_audit.yaml")

    train_config = config["data"]['models']

    random_state = train_config.get("random_state")

    logger.info("Building classification model pipelines.")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=train_config["logistic_regression"].get("max_iter"),
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=train_config["random_forest"].get("n_estimators"),
            random_state=random_state,
        ),
        "svc": SVC(
            kernel=train_config["svc"].get("kernel"),
            probability=train_config["svc"].get("probability"),
            random_state=random_state,
        ),
    }

    models = {}

    for name , model in models.items():
        models[name] = Pipeline(
            [
                ("preprocessor",preprocessor),
                ("model",model)
            ]
        )

    logger.info(
        "Successfully created %d classification pipelines.",
        len(models),
    )

    return models

def train_model(preprocessor: ColumnTransformer,x_train: pd.DataFrame,y_train):

    models = classification_models(preprocessor) 

    if not models:
        raise ModelTrainingError("No models provided for training.")

    fitted_models = {}

    logger.info(
        "Starting training of %d models using %d samples.",
        len(models),
        len(x_train),
    )

    for model_name , model in models.items():
        try:
            logger.info("training model %s",model_name)

            model_fit = clone(model)
            model_fit.fit(x_train,y_train)

            fitted_models[model_name] = model_fit

            logger.info(
                "Model '%s' trained successfully.",
                model_name,
            )

        except Exception as exc:
            logger.exception(
                "Training failed for model '%s'.",
                model_name,
            )

            raise ModelTrainingError(
                f"Failed to train model '{model_name}'."
            ) from exc

    logger.info("All models trained successfully.")

    return fitted_models