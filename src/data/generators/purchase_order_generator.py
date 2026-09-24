from __future__ import annotations

import numpy as np
import pandas as pd


PO_PRIORITIES = ["Normal", "High", "Critical"]

PO_STATUSES = [
    "Open",
    "Partially Delivered",
    "Delivered",
    "Closed",
]


def generate_purchase_orders(
    projects: pd.DataFrame,
    materials: pd.DataFrame,
    suppliers: pd.DataFrame,
    seed: int = 42,
    min_pos_per_project: int = 80,
    max_pos_per_project: int = 220,
) -> pd.DataFrame:
    """
    Generate synthetic purchase orders using existing enterprise master data.

    Relationships:
        Project -> Purchase Order -> Material -> Supplier

    The generator intentionally creates realistic relationships between:
        - project complexity
        - material criticality
        - supplier reliability
        - material price
        - required dates
        - procurement priority
    """

    rng = np.random.default_rng(seed)

    required_project_columns = {
        "project_id",
        "planned_start_date",
        "planned_end_date",
        "project_complexity",
    }

    required_material_columns = {
        "material_id",
        "category",
        "standard_price",
        "criticality",
        "lead_time_days",
        "minimum_order_quantity",
    }

    required_supplier_columns = {
        "supplier_id",
        "supplier_category",
        "supplier_rating",
        "reliability_score",
        "payment_terms_days",
    }

    missing_projects = required_project_columns - set(projects.columns)
    missing_materials = required_material_columns - set(materials.columns)
    missing_suppliers = required_supplier_columns - set(suppliers.columns)

    if missing_projects:
        raise ValueError(
            f"Missing project columns: {sorted(missing_projects)}"
        )

    if missing_materials:
        raise ValueError(
            f"Missing material columns: {sorted(missing_materials)}"
        )

    if missing_suppliers:
        raise ValueError(
            f"Missing supplier columns: {sorted(missing_suppliers)}"
        )

    projects = projects.copy()
    materials = materials.copy()
    suppliers = suppliers.copy()

    projects["planned_start_date"] = pd.to_datetime(
        projects["planned_start_date"]
    )
    projects["planned_end_date"] = pd.to_datetime(
        projects["planned_end_date"]
    )

    records = []
    po_counter = 1

    for _, project in projects.iterrows():

        project_complexity = project["project_complexity"]

        complexity_factor = {
            "LOW": 0.80,
            "MEDIUM": 1.00,
            "HIGH": 1.35,
            "VERY_HIGH": 1.70,
        }.get(project_complexity, 1.00)

        base_po_count = rng.integers(
            min_pos_per_project,
            max_pos_per_project + 1,
        )

        po_count = int(base_po_count * complexity_factor)

        project_materials = materials.sample(
            n=min(po_count, len(materials)),
            replace=True,
            random_state=int(rng.integers(0, 1_000_000)),
        ).reset_index(drop=True)

        for i in range(po_count):

            material = project_materials.iloc[i % len(project_materials)]

            # ---------------------------------------------------------
            # Supplier selection
            # ---------------------------------------------------------

            compatible_suppliers = suppliers[
                suppliers["supplier_category"].isin(
                    [
                        material["category"],
                        "General Materials",
                    ]
                )
            ]

            if compatible_suppliers.empty:
                compatible_suppliers = suppliers

            supplier = compatible_suppliers.sample(
                n=1,
                random_state=int(rng.integers(0, 1_000_000)),
            ).iloc[0]

            # ---------------------------------------------------------
            # Quantity
            # ---------------------------------------------------------

            minimum_quantity = max(
                1,
                int(material["minimum_order_quantity"]),
            )

            quantity_multiplier = rng.integers(1, 15)

            ordered_quantity = (
                minimum_quantity * quantity_multiplier
            )

            # ---------------------------------------------------------
            # Unit price
            # ---------------------------------------------------------

            price_variation = rng.uniform(0.90, 1.15)

            unit_price = round(
                float(material["standard_price"])
                * price_variation,
                2,
            )

            total_value = round(
                ordered_quantity * unit_price,
                2,
            )

            # ---------------------------------------------------------
            # Procurement date
            # ---------------------------------------------------------

            project_start = project["planned_start_date"]
            project_end = project["planned_end_date"]

            project_duration = max(
                30,
                (project_end - project_start).days,
            )

            procurement_offset = int(
                rng.integers(
                    0,
                    max(1, project_duration - 1),
                )
            )

            po_date = project_start + pd.Timedelta(
                days=procurement_offset
            )

            # ---------------------------------------------------------
            # Required date
            # ---------------------------------------------------------

            lead_time = int(material["lead_time_days"])

            safety_buffer = {
                "LOW": 20,
                "MEDIUM": 15,
                "HIGH": 10,
                "CRITICAL": 5,
            }.get(material["criticality"], 15)

            required_date = po_date + pd.Timedelta(
                days=lead_time + safety_buffer
            )

            # Keep required date inside a reasonable project window.
            if required_date > project_end:
                required_date = project_end

            # ---------------------------------------------------------
            # Priority
            # ---------------------------------------------------------

            if material["criticality"] == "CRITICAL":
                priority = "Critical"

            elif material["criticality"] == "HIGH":
                priority = (
                    "High"
                    if rng.random() < 0.70
                    else "Normal"
                )

            else:
                priority = "High" if rng.random() < 0.20 else "Normal"

            # ---------------------------------------------------------
            # PO status
            # ---------------------------------------------------------

            status_probability = rng.random()

            if status_probability < 0.10:
                po_status = "Open"
            elif status_probability < 0.20:
                po_status = "Partially Delivered"
            elif status_probability < 0.75:
                po_status = "Delivered"
            else:
                po_status = "Closed"

            # ---------------------------------------------------------
            # Expected delivery
            # ---------------------------------------------------------

            expected_delivery_date = (
                po_date
                + pd.Timedelta(days=lead_time)
            )

            # Supplier reliability influences expected variation.
            reliability = float(supplier["reliability_score"])

            expected_variation = int(
                np.clip(
                    rng.normal(
                        loc=(1.0 - reliability) * 5,
                        scale=2.0,
                    ),
                    -3,
                    10,
                )
            )

            expected_delivery_date += pd.Timedelta(
                days=expected_variation
            )

            # ---------------------------------------------------------
            # Record
            # ---------------------------------------------------------

            records.append(
                {
                    "po_id": f"PO-{po_counter:07d}",
                    "project_id": project["project_id"],
                    "material_id": material["material_id"],
                    "supplier_id": supplier["supplier_id"],
                    "po_date": po_date,
                    "required_date": required_date,
                    "ordered_quantity": ordered_quantity,
                    "unit_price": unit_price,
                    "total_value": total_value,
                    "payment_terms_days": int(
                        supplier["payment_terms_days"]
                    ),
                    "priority": priority,
                    "po_status": po_status,
                    "expected_delivery_date": expected_delivery_date,
                }
            )

            po_counter += 1

    purchase_orders = pd.DataFrame(records)

    return purchase_orders