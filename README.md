# weather-generator

Stochastic weather generator for point queries at arbitrary `(lat, lon,
datetime)`. Trains on the raw 3-hourly NOAA GFS-Wave / WAVEWATCH III + GFS
wind archive via Tucker compression → Dynamic Mode Decomposition → per-mode
Ornstein–Uhlenbeck processes, with per-cell Gaussian-copula marginals and
harmonic climatology removal.

See [`docs/model_design.md`](docs/model_design.md) for the full design and
[`AGENTS.md`](AGENTS.md) for the source code conventions used in this repo.

## Install

```bash
poetry install
```

## Usage

```python
from pathlib import Path
from datetime import datetime, timezone

from weather import WeatherModel, sample, sample_batch, simulate_field

model = WeatherModel.load(Path("./models/v1"))

value = sample(
    model,
    lat=35.5,
    lon=-40.2,
    t=datetime(2025, 6, 15, 9, tzinfo=timezone.utc),
    variable="sigwh",
    seed=42,
)
```

There is no CLI. Training happens in
[`notebooks/test_pipeline.ipynb`](notebooks/test_pipeline.ipynb); querying a
trained model is programmatic, as above.

## Development

```bash
black src/ tests/
isort src/ tests/
mypy --strict src/
pytest --cov=src
```
