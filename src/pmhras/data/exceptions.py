

class DataAuditError(Exception):
    pass


class DataLoadingError(DataAuditError):
    pass


class DataValidationError(DataAuditError):
    pass


class AuditConfigurationError(DataAuditError):
    pass
