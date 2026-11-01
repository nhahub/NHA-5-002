import pytest
from pydantic import BaseModel
from app.validation import OutputValidator, OutputValidationError

class Item(BaseModel): value: int
def test_output_validator_accepts_json(): assert OutputValidator().validate('{"value": 2}', Item).value == 2
def test_output_validator_rejects_malformed_json():
    with pytest.raises(OutputValidationError): OutputValidator().validate("not json", Item)
