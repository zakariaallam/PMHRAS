import logging

import pandas as pd 

from sklearn.model_selection import train_test_split

from pmhras.preprocessing.settings import DataSplitError

from pmhras.config.settings import load_config 

logger = logging.getLogger(__name__)

def data_split(df: pd.DataFrame):

    logger.info("starting dataset spliting")

    config = load_config("configs/data_audit.yaml")

    split_config = config['preprocessing']['split']
    data_config = config['data']

    if df.empty:
        raise DataSplitError(
            "Cannot split an empty DataFrame."
        )

    if data_config.get("target_column") not in df.columns:
        raise DataSplitError(
            f"Target column {data_config.get("target_column")} does not exist."
        )

    x = df.drop(columns=data_config.get("target_column"))
    y = df[data_config.get("target_column")]

    try:
        x_train , x_test , y_train , y_test = train_test_split(
            x,
            y,
            test_size = split_config.get("test_size"),
            random_state = split_config.get("random_state")
        )

    except ValueError as exc:
        logger.exception("Dataset splitting failed.")

        raise DataSplitError(
            f"Unable to split dataset: {exc}"
        ) from exc

    logger.info("Dataset complited : %d training simples , %d testing simples",len(x_train),len(x_test))

    return x_train , x_test , y_train , y_test
