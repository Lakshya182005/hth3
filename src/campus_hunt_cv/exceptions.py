"""Custom exceptions for the campus location verification pipeline."""

from typing import Optional


class CampusHuntError(Exception):
    """Base exception for all campus hunt verification errors."""

    def __init__(self, message: str, reason_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.reason_code = reason_code or "ERROR_INTERNAL"


class ImageValidationError(CampusHuntError):
    """Raised when an input image fails decoding, integrity, or limit checks."""

    def __init__(self, message: str, reason_code: str = "ERROR_INVALID_IMAGE"):
        super().__init__(message, reason_code=reason_code)


class ReferenceDataError(CampusHuntError):
    """Raised when reference embeddings are missing, corrupt, or insufficient."""

    def __init__(self, message: str, reason_code: str = "ERROR_REFERENCE_DATA"):
        super().__init__(message, reason_code=reason_code)


class ModelInferenceError(CampusHuntError):
    """Raised when model loading, preprocessing, or forward pass fails."""

    def __init__(self, message: str, reason_code: str = "ERROR_MODEL_INFERENCE"):
        super().__init__(message, reason_code=reason_code)


class ConfigurationError(CampusHuntError):
    """Raised when pipeline configuration is invalid or missing required parameters."""

    def __init__(self, message: str, reason_code: str = "ERROR_CONFIGURATION"):
        super().__init__(message, reason_code=reason_code)

