"""Print the deterministic health summary for one FD001 engine."""

import argparse

from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import load_engine_readings
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.health.engine_health import DEFAULT_WINDOW, summarise_engine_health


def main() -> None:
    """Load one engine's readings from PostgreSQL and print its trend summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("unit_id", type=int)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    arguments = parser.parse_args()

    with Session(get_engine()) as session:
        readings = load_engine_readings(session, arguments.split, arguments.unit_id)

    summary = summarise_engine_health(readings, window=arguments.window)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
