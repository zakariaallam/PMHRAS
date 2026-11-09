

class DataAuditError(Exception):
    ""


class DataLoadingError(DataAuditError):
    ""


class DataValidationError(DataAuditError):
    ""


class AuditConfigurationError(DataAuditError):
    ""
