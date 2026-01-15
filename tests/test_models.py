import importlib.util
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("faker") is None:
    pytest.skip("faker dependency is missing", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import make_faker, sample_seed_pack


def test_sample_seed_pack_has_fields() -> None:
    faker = make_faker("en_US", seed=123)
    seed_pack = sample_seed_pack(faker)

    assert seed_pack.job
    assert seed_pack.industry
    assert seed_pack.country
    assert seed_pack.buzzwords
    assert seed_pack.event
    assert seed_pack.forbidden_city
    assert seed_pack.forbidden_company
