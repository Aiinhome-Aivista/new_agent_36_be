"""
API Response Standardization Utility Module for DevKnow.

Provides consistent, production-grade JSON response envelopes across all REST API controllers.
Adheres to enterprise REST design standards:
- `success`: bool (True for 2xx, False for 4xx/5xx)
- `status`: str ("success" | "error")
- `status_code`: int (HTTP status code, e.g., 200, 201, 400, 404, 500)
- `message`: str (Human-readable descriptive message)
- `data`: Optional[Any] (Main response payload)
- Backward-compatibility preservation: merges extra keyword arguments into the root JSON
  so existing frontend clients consuming top-level keys continue to work seamlessly.
"""

from typing import Any, Optional, Dict, Tuple
from flask import jsonify, Response


def success_response(
    data: Optional[Any] = None,
    message: str = "Operation completed successfully",
    status_code: int = 200,
    **extra
) -> Tuple[Response, int]:
    """
    Constructs a standardized success response envelope.

    Args:
        data: Optional primary response payload (e.g. list, dictionary, object).
        message: Descriptive summary of the successful action.
        status_code: HTTP status code (default: 200).
        **extra: Additional domain-specific fields to include at the root JSON level
                 for backward compatibility with frontend consumers.

    Returns:
        Tuple[Response, int]: Flask JSON response and HTTP status code tuple.
    """
    payload: Dict[str, Any] = {
        "success": True,
        "status": "success",
        "status_code": status_code,
        "message": message,
    }
    if data is not None:
        payload["data"] = data

    # Merge domain-specific fields for 100% backward compatibility
    payload.update(extra)
    return jsonify(payload), status_code


def error_response(
    message: str = "An error occurred",
    status_code: int = 400,
    error: Optional[str] = None,
    data: Optional[Any] = None,
    **extra
) -> Tuple[Response, int]:
    """
    Constructs a standardized error response envelope.

    Args:
        message: Human-readable error description.
        status_code: HTTP error status code (e.g., 400, 401, 403, 404, 500).
        error: Specific machine/exception error message (defaults to message).
        data: Optional diagnostic data or context.
        **extra: Additional fields to include in the root JSON.

    Returns:
        Tuple[Response, int]: Flask JSON response and HTTP status code tuple.
    """
    err_text = error or message
    payload: Dict[str, Any] = {
        "success": False,
        "status": "error",
        "status_code": status_code,
        "message": message,
        "error": err_text,
    }
    if data is not None:
        payload["data"] = data

    payload.update(extra)
    return jsonify(payload), status_code
