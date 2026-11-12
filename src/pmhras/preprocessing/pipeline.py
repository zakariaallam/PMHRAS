import logging 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,OrdinalEncoder,StandardScaler)

from pmhras.preprocessing.settings import (FeatureValidationError,OrdinalConfigurationError)

logger = logging.getLogger(__name__)

def validation_feature(
    numerical_features: list[str],
    ordinal_features: list[str],
    nominal_features: list[str],
) -> None:

    feature_group = {
         "numerical": numerical_features,
        "ordinal": ordinal_features,
        "nominal": nominal_features,
    }

    for group_name , feature in feature_group.items():

        if not isinstance(feature,list):
            raise FeatureValidationError(f"{group_name} feature must be a list")

        if len(feature) != len(set(feature)):
            raise FeatureValidationError(f"duplicate columns found in {group_name} features")

    all_features = (numerical_features + ordinal_features + nominal_features)

    if not all_features:
        raise FeatureValidationError("all features is empty")

    logger.info("Feature validation successful %d numerical, "
        "%d ordinal, %d nominal features.",len(numerical_features),len(ordinal_features),len(nominal_features))

def validation_ordinal(ordinal_features: list[str],ordinal_categories: list[list[str]]):

    if not ordinal_features:
        raise OrdinalConfigurationError(" Ordinal feature is empty")

    if not ordinal_categories:
        raise OrdinalConfigurationError("Ordinal categorie is empty")

    for categories in ordinal_categories:

        if len(categories) != len(set(categories)):
            raise OrdinalConfigurationError("Ordinal categories must not contain duplicated")

    logger.info("ordinal categorie validation successful for %d feature",len(ordinal_categories))


def preprocessing(
    numerical_features: list[str],
    ordinal_features: list[str],
    nominal_features: list[str],
    ordinal_categories: list[list[str]] 
):
    logger.info("starting preprocessing configuration")

    validation_feature(numerical_features,ordinal_features,nominal_features)

    validation_ordinal(ordinal_features,ordinal_categories)

    tranformers = []

    if numerical_features:
        numerical_pipeline = (
            [
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler" , StandardScaler())
            ]
        )

        tranformers.append(("numerical",numerical_pipeline,numerical_features))

    if ordinal_features:
        ordinal_pipeline = (
            [
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OrdinalEncoder(categories=ordinal_categories,handle_unknown="use_encoded_value",unknown_value=1))
            ]
        )

        tranformers.append(("ordinal",ordinal_pipeline,ordinal_features))

    if nominal_features:
        nominal_pipeline = (
            [
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        tranformers.append(("nominal",nominal_pipeline,nominal_features))

    preprocessor = ColumnTransformer(
        transformers=tranformers,
        remainder="drop"
    )

    logger.info("Preprocessing Configuration successfully ")

    return preprocessor

