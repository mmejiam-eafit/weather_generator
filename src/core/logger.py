"""The package's one structured logger (see AGENTS.md *Logging*).

`weather` is a library consumed programmatically, so this module routes
through the standard library (`structlog.stdlib.LoggerFactory`) and never
configures handlers: an application owns its logging setup, and a library
that attaches root handlers at import time hijacks its consumer's output.
The training notebook configures output, e.g.
``logging.basicConfig(level=logging.INFO)``. Unconfigured, records below
WARNING are dropped by the standard library's defaults.

`structlog.wrap_logger` binds the processor chain to this one logger
instead of calling `structlog.configure`, which would change structlog's
global defaults for every other library in the process.
"""

from typing import Final

import structlog

LOGGER_NAME: Final[str] = "weather_generator"

logger: Final[structlog.stdlib.BoundLogger] = structlog.wrap_logger(
    structlog.stdlib.LoggerFactory()(LOGGER_NAME),
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.KeyValueRenderer(key_order=["event"], sort_keys=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
)
