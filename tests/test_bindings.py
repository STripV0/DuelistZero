"""
Tests for the core bindings (OcgCore).
"""

from pathlib import Path
import pytest

from duelist_zero.core.bindings import OcgCore
from duelist_zero.core.callbacks import CallbackManager

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LIB = _PROJECT_ROOT / "lib" / "libocgcore.so"
_DB = _PROJECT_ROOT / "data" / "cards.cdb"
_SCRIPTS = _PROJECT_ROOT / "data" / "script"


@pytest.fixture(scope="module")
def core():
    return OcgCore(_LIB)


@pytest.fixture(scope="module")
def core_with_cb(core):
    cb = CallbackManager(core, _DB, _SCRIPTS)
    cb.register()
    return core


class TestLibraryLoads:
    def test_library_loads(self, core):
        assert core is not None
        assert core.lib is not None

    def test_create_duel(self, core_with_cb):
        pduel = core_with_cb.create_duel(12345)
        assert pduel != 0, "Expected non-zero duel handle"
        core_with_cb.end_duel(pduel)

    def test_set_player_info(self, core_with_cb):
        pduel = core_with_cb.create_duel(99)
        # Should not raise
        core_with_cb.set_player_info(pduel, 0, 8000, 5, 1)
        core_with_cb.set_player_info(pduel, 1, 8000, 5, 1)
        core_with_cb.end_duel(pduel)

    def test_process_returns_int(self, core_with_cb):
        pduel = core_with_cb.create_duel(7)
        core_with_cb.set_player_info(pduel, 0, 8000, 5, 1)
        core_with_cb.set_player_info(pduel, 1, 8000, 5, 1)
        result = core_with_cb.process(pduel)
        assert isinstance(result, int)
        core_with_cb.end_duel(pduel)
