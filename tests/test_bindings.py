from __future__ import annotations

from wesi.bindings.build_ffi import get_cdef


def test_cdef_exposes_required_symbols() -> None:
    cdef = get_cdef()
    assert "wesi_grid_t" in cdef
    assert "wesi_submodel_t" in cdef
    assert "wesi_shot_t" in cdef
    assert "wesi_run_forward" in cdef
    assert "wesi_run_rtm" in cdef
