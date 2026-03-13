from __future__ import annotations

from math import isclose
from pathlib import Path
import re
import subprocess

from wesi.domain import SurveyGrid
from wesi.domain.survey_grid import GridControlPoint


def _assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
    assert isclose(actual, expected, rel_tol=tol, abs_tol=tol), f"expected {expected}, got {actual}"


def _assert_pair_close(actual: tuple[float, float], expected: tuple[float, float], tol: float = 1e-6) -> None:
    _assert_close(actual[0], expected[0], tol)
    _assert_close(actual[1], expected[1], tol)


def _assert_rel_close(actual: float, expected: float, rel_tol: float = 1.0e-4) -> None:
    rel_err = abs(actual - expected) / abs(expected)
    print(f"check: actual={actual:.12f}, expected={expected:.12f}, rel_err={rel_err:.12e}")
    assert rel_err < rel_tol, f"relative error {rel_err} >= {rel_tol}, actual={actual}, expected={expected}"


def _extract_named_float(text: str, name: str) -> float:
    match = re.search(rf"{name}=([-+]?\d+(?:\.\d+)?)", text)
    assert match is not None, f"could not find {name}=... in output:\n{text}"
    return float(match.group(1))


def test_survey_grid_projection_with_field_case() -> None:
    point0 = GridControlPoint("Point0", (444827.0, 4212368.0), 1.0, 1.0)
    point1 = GridControlPoint("Point1", (478746.0, 4203759.0), 1.0, 7000.0)
    point3 = GridControlPoint("Point3", (445073.0, 4213337.0), 3.0, 1.0)
    target_rw = (602389.0, 4223447.0)
    target_grid = (100.0, 30000.0)

    print("build grid from Point0, Point1, Point3")
    grid = SurveyGrid.from_three_points(point0=point0, point1=point1, point3=point3)

    print(f"step_inl = {grid.step_inl:.12f}")
    print(f"step_cxl = {grid.step_cxl:.12f}")
    print(f"inl_angle_deg = {grid.inl_angle_deg:.12f}")
    print(f"cxl_angle_deg = {grid.cxl_angle_deg:.12f}")
    print(f"point2 = rw_x={grid.point2.rw_x:.12f}, rw_y={grid.point2.rw_y:.12f}, inl={grid.point2.inl:.12f}, cxl={grid.point2.cxl:.12f}")

    calc_inl, calc_cxl = grid.grid_from_rw(target_rw)
    print(f"grid_from_rw({target_rw[0]:.6f}, {target_rw[1]:.6f}) -> inl={calc_inl:.12f}, cxl={calc_cxl:.12f}")
    _assert_rel_close(calc_inl, target_grid[0], rel_tol=1.0e-2)
    _assert_rel_close(calc_cxl, target_grid[1], rel_tol=1.0e-2)

    calc_rw_x, calc_rw_y = grid.rw_from_grid(target_grid[0], target_grid[1])
    print(f"rw_from_grid({target_grid[0]:.6f}, {target_grid[1]:.6f}) -> rw_x={calc_rw_x:.12f}, rw_y={calc_rw_y:.12f}")
    _assert_rel_close(calc_rw_x, target_rw[0])
    _assert_rel_close(calc_rw_y, target_rw[1])

    print("check angle")
    _assert_rel_close(grid.inl_angle_deg, 14.24, rel_tol=1.0e-3)


def test_survey_grid_json_round_trip(tmp_path: Path | None = None) -> None:
    grid = SurveyGrid.from_three_points(
        point0=GridControlPoint("Point0", (444827.0, 4212368.0), 1.0, 1.0),
        point1=GridControlPoint("Point1", (478746.0, 4203759.0), 1.0, 7000.0),
        point3=GridControlPoint("Point3", (445073.0, 4213337.0), 3.0, 1.0),
    )

    base = tmp_path if tmp_path is not None else Path(".tmp_test_survey_grid")
    base.mkdir(parents=True, exist_ok=True)
    path = base / "survey-grid.json"
    restored = SurveyGrid.from_json_file(grid.to_json_file(path))

    assert restored.as_dict() == grid.as_dict()
    _assert_close(restored.step_inl, grid.step_inl)
    _assert_close(restored.step_cxl, grid.step_cxl)


def test_survey_grid_summary_contains_projection_info() -> None:
    grid = SurveyGrid.from_three_points(
        point0=GridControlPoint("Point0", (444827.0, 4212368.0), 1.0, 1.0),
        point1=GridControlPoint("Point1", (478746.0, 4203759.0), 1.0, 7000.0),
        point3=GridControlPoint("Point3", (445073.0, 4213337.0), 3.0, 1.0),
    )

    summary = grid.summary_text()
    assert "Point0->Point3 defines inl direction" in summary
    assert "inl_unit" in summary
    assert "cxl_unit" in summary
    assert "rw_x=" in summary