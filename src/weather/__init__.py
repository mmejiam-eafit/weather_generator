"""weather — public API for the stochastic weather generator.

(see docs/model_design.md §8.4)

Depends on `core` and `train`. Consumers should import only from here.
"""

from weather.model import WeatherModel
from weather.simulate import sample, sample_batch, simulate_field
from weather.validate import full_report

__all__ = [
    "WeatherModel",
    "sample",
    "sample_batch",
    "simulate_field",
    "full_report",
]
