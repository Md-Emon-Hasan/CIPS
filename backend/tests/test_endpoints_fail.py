import importlib
from unittest.mock import patch

import backend.app.api.endpoints as endpoints


def test_model_load_failure():
    # Force an exception during open() in endpoints.py
    # We use importlib.reload to re-execute the module-level try-except block
    with patch("builtins.open", side_effect=Exception("Force load failure")):
        importlib.reload(endpoints)
        assert endpoints.pipe is None


def test_model_load_success():
    # Test successful reload (optional, but good for stability)
    with patch("builtins.open"):
        with patch("pickle.load", return_value="mock_pipe"):
            importlib.reload(endpoints)
            assert endpoints.pipe == "mock_pipe"
