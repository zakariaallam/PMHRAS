import joblib
from pathlib import Path

def save_model(
    model,
    config
) -> Path:

    data_config = config['data']
    output_path = data_config.gat("model_path_save")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return path