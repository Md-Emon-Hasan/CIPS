from unittest.mock import MagicMock, patch

import pytest
from backend.app.db import get_session, init_db


@patch("backend.app.db.SQLModel.metadata.create_all")
@patch("backend.app.db.engine")
def test_init_db(mock_engine, mock_create_all):
    init_db()
    mock_create_all.assert_called_once_with(mock_engine)


def test_get_session():
    # Helper to test get_session generator
    # We can mock Session
    with patch("backend.app.db.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        gen = get_session()
        session = next(gen)
        assert session == mock_session

        # Test cleaning up
        with pytest.raises(StopIteration):
            next(gen)
