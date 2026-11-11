import numpy as np
import pandas as pd
import logging
from typing import Any
from pathlib import Path
import json
import argparse

from pmhras.config.logging_config import (configure_logging)

from pmhras.config.settings import load_config

from pmhras.data.exceptions import (
    DataAuditError,
    DataLoadingError,
    DataValidationError
)

logger = logging.getLogger(__name__)

def load_dataset(config: dict[str,Any]):
    data_config = config['data']
    dataset_path = Path(data_config.get("raw_path"))

    logger.info("Loading Dataset %s ",dataset_path)

    if not dataset_path.exists():
        raise DataLoadingError(
            f"Dataset file not found: {dataset_path}"
        )

    try:
        df = pd.read_csv(
            dataset_path,
            encoding=data_config.get('encoding','utf-8'),
            sep= data_config.get('separator',','),
            )
    except DataLoadingError as exc:
        raise  DataLoadingError(
            f"unable to read dataset {dataset_path}"
        ) from exc

    if df.empty:
        raise DataValidationError("dataset is empty")

    if df.columns.duplicated().any():
        duplicated_colums = df.columns[
            df.columns.duplicated()
        ].to_list

        raise DataValidationError(
            f"duplicate columns names : {duplicated_colums}"
        )

    logger.info("data loading successfuly, %d rows , %d columns",*df.shape)

    return df

def inspect_structure(df: pd.DataFrame) -> dict[str,Any]:

    logger.info("inspecting dataset structure")

    return {
        "rows" : len(df),
        "columns_count" : len(df.columns),
        "columns" : df.columns.to_list(),
        "data_type" : df.dtypes.astype(str).to_dict(),
        # "data_describe" : df.describe().to_dict(),
        "preview" : df.head().to_dict(orient="records")
    }

def analize_missing_values(df: pd.DataFrame) -> dict[str,Any]:

    logger.info("Analizing Missing Value")

    missing_values = df.isna().sum()

    missing_porcentage = (missing_values / len(df) * 100).round(2)

    result = pd.DataFrame(
        {
        "missing_values" : missing_values,
        "missing_porcentage" : missing_porcentage
        }
    )

    result = result[result["missing_values"] > 0]

    return {
        "total_missing_values" : int(missing_values.sum()),
        "numbre_columns_with_missing_values" : int((missing_values > 0).sum()),
        "details" : result.to_dict(orient="index")
    }

def analize_duplicate(df: pd.DataFrame) -> dict[str,Any]:

    logger.info("ckecking duplicated row")

    duplicates = df.duplicated().sum()

    return {
        "duplicate_row" : int(duplicates),
        "duplicate_porcentage" : (duplicates / len(df) *100).round(2)
    }

def incoherences(df: pd.DataFrame):

    logger.info("checking incoherences")

    all_columns_null = df.columns[df.isna().all()].to_list()
    columns_constant = df.columns[df.nunique() <= 1].to_list()
    number_values_inlogic_between_media_screen = len(df[df['Social_Media_Hours'] > df['Screen_Time_Hours']])

    string_vide = {}

    for column in df.select_dtypes(include=["string","object"]).columns:
        count = df[column].str.strip().eq("").sum()

        if count > 0:
            string_vide[column] = count

    return {
        "all_columns_null" : all_columns_null,
        "columns_constant" : columns_constant,
        "number_values_intogic_between_media_screen": number_values_inlogic_between_media_screen
    }

def analyze_outliers(df: pd.DataFrame , config) -> dict[str,Any]:

    logger.info("Analisation outliers")
    audit_outlier = config['audit']
    outlire_config = audit_outlier.get("outliers",{})
    method = outlire_config.get("method","iqr")

    if method != "iqr":
        raise DataValidationError(
            f"method outlier is incorecte {method}"
        )

    multiplier = outlire_config.get("multiplier",1.5)

    columns_numeric = df.select_dtypes(include=np.number).columns
    result = {}

    for column in columns_numeric:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        IQ1 = Q1 - multiplier * IQR
        IQ3 = Q3 + multiplier * IQR

        outlier_mask = (df[column] < IQ1) | (df[column] > IQ3)

        number_outlier = int(outlier_mask.sum())

        result[column] = {
            "Q1" : int(Q1),
            "Q3" : int(Q3),
            "IQR" : int(IQR),
            "IQ1" : int(IQ1),
            "IQ3" : int(IQ3),
            "number_outlier" : number_outlier,
            "outlier_pourcentage" : round((number_outlier / len(df[column]) * 100 ),2)
        }
    return result

def statistiques_descriptives(df: pd.DataFrame) -> dict[str,Any]:

    logger.info("Generation statistiques descriptives")

    numeric = df.select_dtypes(include=np.number)

    categorie = df.select_dtypes(include=["string","object","bool","category"])

    numeric_stat = (numeric.describe().to_dict())

    categorie_stat = (categorie.describe().to_dict())

    return {
        "numerical" : numeric_stat,
        "categorical" : categorie_stat
    }

def analyze_target(df: pd.DataFrame,config: dict[str,Any]) -> dict[str,Any]:

    target = config['data'].get("target_column")

    if not target:
        logger.warning("target column not configured")

    if target not in df.columns:
        raise DataValidationError(f"target name {target} not in dataset ")

    logger.info("Analizing target column ")

    target_count = df[target].value_counts(dropna=False)

    distribution = {}

    for key , value in target_count.items():
        distribution[str(key)] = {
                "count" : int(value),
                "percentage" : round((value / len(df) * 100),2),
            }

    result = {
        "target" : target,
        "number_class" : int(df[target].nunique()),
        "distribution" : distribution
    }

    return result


def analize_correlation(df: pd.DataFrame , config: dict[str,Any]) -> dict[str,Any]:

    logger.info("Calculation numeric correlation")

    excluded = config["data"].get("id_columns")

    target = config["data"].get("target_column")

    if target:
        excluded.append(target)

    numeric_df = df.select_dtypes(include=np.number).drop(columns=list(excluded) , errors="ignore")

    correlation = numeric_df.corr(method="spearman")

    return {
        "correlation" : correlation.to_dict()
    }


def run_audit(config_path: str ) -> dict[str,Any]:

    config = load_config(config_path)
    loggin_config  = config["logging"]

    configure_logging(
        log_dir=loggin_config.get("directory"),
        filename= loggin_config.get("filename"),
        level=loggin_config.get("level")
    )

    logger.info("Starting data Audit")

    df = load_dataset(config)
    audit_config = config["audit"]

    report = {
        "project" : config["project"].get("name"),
        "dataset" : str(config['data'].get("raw_path")),
        "structure" : inspect_structure(df)
    }

    if audit_config.get("check_missing_values"):
        report["missing_values"] = analize_missing_values(df)

    if audit_config.get('check_duplicates'):
        report['duplicates_value'] = analize_duplicate(df)

    if audit_config.get('check_outliers'):
        report["outliers"] = analyze_outliers(df,config)

    if audit_config.get('descriptive_statistics'):
        report["statistiques_descriptives"] = statistiques_descriptives(df)

    if audit_config.get('target_distribution'):
        report['target_analysis'] = analyze_target(df,config)

    if audit_config.get('calculate_correlations'):
        report['correlation'] = analize_correlation(df,config)

    report_config = config["report"]

    if report_config.get("save_json"):

        output_dir = Path(report_config.get("output_dir"))

        output_dir.mkdir(parents=True,exist_ok=True)

        output_path = output_dir / report_config.get("filename")

        with output_path.open("w",encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=True
            )
        logger.info(f"audit report saved in {output_path}")

    logger.info("data audit completed successfully")

    return report

def main() -> None:

    parser = argparse.ArgumentParser(
        description= "Prédiction du risque de santé mentale chez les étudiants"
    )

    parser.add_argument(
        "--config",
        default="configs/data_audit.yaml",
        help="Path to the YAML configuration."
    )

    args = parser.parse_args()

    try:
        run_audit(args.config)

    except DataValidationError:
        logger.exception("Data Audit fieled")
        raise SystemExit(1) from None

    except Exception:
        logger.exception("Error unixpected during data audit")
        raise SystemExit(1) from None

if __name__ == "__main__":
    main()