import pandas as pd

from src.data.generators.material_generator import (
    generate_materials,
)

from src.data.generators.supplier_generator import (
    generate_suppliers,
)

from src.data.generators.employee_generator import (
    generate_employees,
)

from src.data.generators.equipment_generator import (
    generate_equipment,
)

from src.data.generators.project_generator import (
    generate_projects,
)

def test_material_generation():

    df = generate_materials(
        n=100,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df["material_id"].is_unique
    assert df["material_code"].is_unique


def test_supplier_generation():

    df = generate_suppliers(
        n=100,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df["supplier_id"].is_unique

    assert df["supplier_rating"].between(
        1,
        5,
    ).all()

    assert df["reliability_score"].between(
        0,
        1,
    ).all()


def test_employee_generation():

    df = generate_employees(
        n=100,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df["employee_id"].is_unique

    assert (
        df["experience_years"] >= 1
    ).all()


def test_equipment_generation():

    df = generate_equipment(
        n=100,
        seed=42,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df["equipment_id"].is_unique

    assert df["operating_hours"].ge(0).all()

    assert df["downtime_hours"].ge(0).all()

    assert df["utilization_rate"].between(
        0,
        1,
    ).all()


def test_project_generation():

    employee_ids = [
        f"EMP-{i:05d}"
        for i in range(1, 101)
    ]

    df = generate_projects(
        n=100,
        seed=42,
        employee_ids=employee_ids,
    )

    assert isinstance(
        df,
        pd.DataFrame,
    )

    assert len(df) == 100

    assert df[
        "project_id"
    ].is_unique

    assert df[
        "contract_value"
    ].gt(0).all()

    assert df[
        "planned_duration_days"
    ].gt(0).all()

    assert df[
        "actual_duration_days"
    ].gt(0).all()

    assert df[
        "delay_flag"
    ].isin([0, 1]).all()

    assert df[
        "project_manager_id"
    ].isin(employee_ids).all()