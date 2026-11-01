from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

class OutputValidationError(ValueError):
    """Raised when provider output cannot be safely converted to the requested schema."""

class OutputValidator:
    def validate(self, payload: dict | str, schema: Type[T]) -> T:
        # Parse provider text first, then enforce the same typed contract used by API responses.
        if isinstance(payload, str):
            import json
            try: payload = json.loads(payload)
            except json.JSONDecodeError as exc: raise OutputValidationError("Provider returned invalid JSON") from exc
        try: return schema.model_validate(payload)
        except ValidationError as exc: raise OutputValidationError(str(exc)) from exc
