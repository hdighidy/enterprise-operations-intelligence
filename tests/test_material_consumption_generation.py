import pandas as pd

from src.data.generators.material_consumption_generator import (
    generate_material_consumption,
)

from src.data.generators.material_generator import (
    generate_materials,
)

from src.data.generators.project_generator import (
    generate_projects,
)

from src.data.generators.activity_generator import (
    generate_project_activities,
)

from src.data.generators.purchase_order_generator import (
    generate_purchase_orders,
)

from src.data.generators.supplier_generator import (
    generate_suppliers,
)

from src.data.generators.delivery_generator import (
    generate_deliveries,
)


def test_material_consumption_generation():

    projects = generate_projects(
        n=3,
        seed=42,
    )

    materials = generate_materials(
        n=20,
        seed=42,
    )

    suppliers = generate_suppliers(
        n=20,
        seed=42,
    )

    activities = generate_project_activities(
        projects=projects,
        seed=42,
    )

    purchase_orders = generate_purchase_orders(
        projects=projects,
        materials=materials,
        suppliers=suppliers,
        seed=42,
        min_pos_per_project=5,
        max_pos_per_project=10,
    )

    deliveries = generate_deliveries(
        purchase_orders=purchase_orders,
        suppliers=suppliers,
        materials=materials,
        seed=42,
    )

    consumption = generate_material_consumption(
        projects=projects,
        materials=materials,
        activities=activities,
        deliveries=deliveries,
        seed=42,
    )

    assert isinstance(
        consumption,
        pd.DataFrame,
    )

    assert not consumption.empty

    assert consumption[
        "consumption_id"
    ].is_unique

    assert consumption[
        "project_id"
    ].isin(
        projects["project_id"]
    ).all()

    assert consumption[
        "activity_id"
    ].isin(
        activities["activity_id"]
    ).all()

    assert consumption[
        "material_id"
    ].isin(
        materials["material_id"]
    ).all()

    assert (
        consumption[
            "planned_daily_quantity"
        ] >= 0
    ).all()

    assert (
        consumption[
            "actual_daily_quantity"
        ] >= 0
    ).all()

    assert consumption[
        "stockout_flag"
    ].isin([0, 1]).all()