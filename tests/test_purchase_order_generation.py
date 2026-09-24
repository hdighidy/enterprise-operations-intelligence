import pandas as pd

from src.data.generators.purchase_order_generator import (
    generate_purchase_orders,
)


def test_purchase_order_generation(projects, materials, suppliers):

    purchase_orders = generate_purchase_orders(
        projects=projects,
        materials=materials,
        suppliers=suppliers,
        seed=42,
        min_pos_per_project=10,
        max_pos_per_project=20,
    )

    assert isinstance(purchase_orders, pd.DataFrame)

    assert not purchase_orders.empty

    assert purchase_orders["po_id"].is_unique

    assert purchase_orders["project_id"].isin(
        projects["project_id"]
    ).all()

    assert purchase_orders["material_id"].isin(
        materials["material_id"]
    ).all()

    assert purchase_orders["supplier_id"].isin(
        suppliers["supplier_id"]
    ).all()

    assert (purchase_orders["ordered_quantity"] > 0).all()

    assert (purchase_orders["unit_price"] > 0).all()

    assert (purchase_orders["total_value"] > 0).all()

    assert purchase_orders["priority"].isin(
        ["Normal", "High", "Critical"]
    ).all()

    assert purchase_orders["po_status"].isin(
        [
            "Open",
            "Partially Delivered",
            "Delivered",
            "Closed",
        ]
    ).all()