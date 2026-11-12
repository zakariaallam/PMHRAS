import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pmhras.models.exceptions import ModelEvaluationError

logger = logging.getLogger(__name__)


def evaluate_predictions(fitted_models:dict[str, Pipeline],x_test: pd.DataFrame,y_test):

    logger.info(
        "Evaluating predictions all modeles"
    )

    for model_name , model in fitted_models.items():

        logger.info(
                "Evaluating predictions ou %s",model_name
            )

        try:

            y_pred = model.predict(x_test)

            results = {
                "accuracy": float(
                    accuracy_score(y_test, y_pred)
                ),
                "precision_macro": float(
                    precision_score(
                        y_test,
                        y_pred,

                    )
                ),
                "recall_macro": float(
                    recall_score(
                        y_test,
                        y_pred,
                    )
                ),
                "f1_macro": float(
                    f1_score(
                        y_test,
                        y_pred,
                    )
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    y_pred,
                ).tolist(),

            }

            logger.info(
                "Evaluation completed. Accuracy=%.4f, F1-macro=%.4f",
                results["accuracy"],
                results["f1_macro"],
            )

            return results

        except ModelEvaluationError:
            raise

        except ValueError as exc:
            logger.exception("Model evaluation failed.")

            raise ModelEvaluationError(
                "Unable to calculate classification metrics."
            ) from exc

