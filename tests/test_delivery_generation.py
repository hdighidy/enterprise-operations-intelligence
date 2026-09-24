import pandas as pd

from src.data.generators.delivery_generator import (
    generate_deliveries,
)

from src.data.generators.material_generator import (
    generate_materials,
)

from src.data.generators.project_generator import (
    generate_projects,
)

from src.data.generators.purchase_order_generator import (
    generate_purchase_orders,
)

from src.data.generators.supplier_generator import (
    generate_suppliers,
)


def test_delivery_generation():

    projects = generate_projects(
        n=5,
        seed=42,
    )

    materials = generate_materials(
        n=50,
        seed=42,
    )

    suppliers = generate_suppliers(
        n=50,
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

    assert isinstance(
        deliveries,
        pd.DataFrame,
    )

    assert not deliveries.empty

    assert deliveries[
        "delivery_id"
    ].is_unique

    assert deliveries[
        "po_id"
    ].isin(
        purchase_orders["po_id"]
    ).all()

    assert deliveries[
        "project_id"
    ].isin(
        projects["project_id"]
    ).all()

    assert deliveries[
        "supplier_id"
    ].isin(
        suppliers["supplier_id"]
    ).all()

    assert deliveries[
        "material_id"
    ].isin(
        materials["material_id"]
    ).all()

    assert (
        deliveries[
            "ordered_quantity"
        ] > 0
    ).all()

    assert (
        deliveries[
            "delivered_quantity"
        ] >= 0
    ).all()

    assert (
        deliveries[
            "accepted_quantity"
        ] >= 0
    ).all()

    assert (
        deliveries[
            "rejected_quantity"
        ] >= 0
    ).all()

    assert deliveries[
        "late_delivery_flag"
    ].isin([0, 1]).all()

    assert deliveries[
        "quality_issue_flag"
    ].isin([0, 1]).all()

    assert (
        deliveries["accepted_quantity"]
        + deliveries["rejected_quantity"]
        <= deliveries["delivered_quantity"] + 0.01
    ).all()