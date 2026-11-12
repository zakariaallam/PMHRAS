import logging

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from pmhras.models.exceptions import ModelOptimizationError

logger = logging.getLogger(__name__)

def CV(config):
    config("configs/data_audit.yaml")
    cv_config = config['cv']

    cv = StratifiedKFold(
        n_splits= cv_config.get("n_splits"),
        shuffle= cv_config.get("shuffle"),
        random_state= cv_config.get("random_state")
    )

    return cv

def optimize_models(models: dict[str,Pipeline],x_train: pd.DataFrame, y_train,config):

    if not models:
        raise ModelOptimizationError("No models provided.")

    if len(x_train) != len(y_train):
        raise ModelOptimizationError(
            "X_train and y_train must have equal lengths."
        )

    cv = CV(config)

    grid_config = config['grid']

    PARAMETER_GRIDS = grid_config.get("parametre_grids")

    optimized_models = {}
    grid_results = {}
    summary_rows = []

    for model_name , model in models.items():
        try:
            if model_name not in PARAMETER_GRIDS:
                raise ModelOptimizationError(
                    f"No parameter grid defined for '{model_name}'."
                )

            logger.info(
                "Optimizing model '%s' with GridSearchCV.",
                model_name,
            )

            grid = GridSearchCV(
                estimator=model,
                param_grid=PARAMETER_GRIDS[model],
                cv=cv,
                scoring=grid_config.get("scoring"),
                refit=grid_config.get("refit"),
                return_train_score=grid_config.get("return_train_score"),
                error_score=grid_config.get("error_score")
            )

            grid.fit(x_train,y_train)

            optimized_models[model_name] = grid.best_estimator_
            grid_results[model_name] = grid

            summary_rows.append(
                {
                    "model" : model_name,
                    "best_cv_f1_macro" : grid.best_score_,
                    "best_params" : grid.best_params_
                }
            )

            logger.info(
                "Model '%s' optimized. Best CV F1: %.4f",
                model_name,
                grid.best_score_,
            )

        except Exception as exc:
            logger.exception(
                "Optimization failed for model '%s'.",
                model_name,
            )
            raise ModelOptimizationError(
                f"Failed to optimize model '{model_name}'."
            ) from exc

    summary = pd.DataFrame(summary_rows).sort_values(by="best_cv_f1_macro",ascending=False)

    return optimized_models , summary , grid_results 