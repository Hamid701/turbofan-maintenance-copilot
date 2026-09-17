"""A pooled connection that the server has closed is replaced, not handed out.

Neon's scale to zero suspends an idle database after five minutes and closes its
open connections with "terminating connection due to administrator command".
``pg_terminate_backend`` raises exactly that error, so this reproduces the
failure against the local database without waiting.
"""

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import NullPool

pytestmark = pytest.mark.integration


def test_a_connection_closed_by_the_server_is_replaced(db_engine: Engine) -> None:
    # Start from an empty pool so the next checkout can only be the closed connection.
    db_engine.dispose()
    with db_engine.connect() as connection:
        closed_pid = connection.execute(text("select pg_backend_pid()")).scalar()

    # The connection is idle in the pool. Close it from the server side through a
    # separate, unpooled connection, as Neon does when it suspends.
    outsider = create_engine(db_engine.url, poolclass=NullPool)
    try:
        with outsider.connect() as connection:
            connection.execute(text("select pg_terminate_backend(:pid)"), {"pid": closed_pid})
    finally:
        outsider.dispose()

    with db_engine.connect() as connection:
        new_pid = connection.execute(text("select pg_backend_pid()")).scalar()

    assert new_pid != closed_pid
