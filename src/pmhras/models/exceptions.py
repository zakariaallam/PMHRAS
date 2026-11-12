class ModelError(Exception):
    pass

class ModelTrainingError(ModelError):
    pass

class ModelComparisonError(ModelError):
    pass

class ModelEvaluationError(ModelError):
    pass

class ModelOptimizationError(ModelError):
    pass