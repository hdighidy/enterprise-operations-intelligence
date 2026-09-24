from __future__ import annotations

import numpy as np
import pandas as pd


def generate_deliveries(
    purchase_orders: pd.DataFrame,
    suppliers: pd.DataFrame,
    materials: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate delivery records from purchase orders.

    Delivery behavior is influenced by:
        - supplier reliability
        - supplier rating
        - material criticality
        - material lead time
        - procurement priority

    The generated dataset is designed to support future
    supplier late-delivery prediction.
    """

    rng = np.random.default_rng(seed)

    required_po_columns = {
        "po_id",
        "project_id",
        "material_id",
        "supplier_id",
        "po_date",
        "required_date",
        "ordered_quantity",
        "priority",
        "expected_delivery_date",
    }

    required_supplier_columns = {
        "supplier_id",
        "supplier_rating",
        "reliability_score",
    }

    required_material_columns = {
        "material_id",
        "criticality",
        "lead_time_days",
    }

    missing_po = (
        required_po_columns
        - set(purchase_orders.columns)
    )

    missing_supplier = (
        required_supplier_columns
        - set(suppliers.columns)
    )

    missing_material = (
        required_material_columns
        - set(materials.columns)
    )

    if missing_po:
        raise ValueError(
            f"Missing purchase-order columns: "
            f"{sorted(missing_po)}"
        )

    if missing_supplier:
        raise ValueError(
            f"Missing supplier columns: "
            f"{sorted(missing_supplier)}"
        )

    if missing_material:
        raise ValueError(
            f"Missing material columns: "
            f"{sorted(missing_material)}"
        )

    po = purchase_orders.copy()
    supplier = suppliers.copy()
    material = materials.copy()

    # ---------------------------------------------------------
    # Date conversion
    # ---------------------------------------------------------

    po["po_date"] = pd.to_datetime(
        po["po_date"]
    )

    po["required_date"] = pd.to_datetime(
        po["required_date"]
    )

    po["expected_delivery_date"] = pd.to_datetime(
        po["expected_delivery_date"]
    )

    # ---------------------------------------------------------
    # Merge supplier information
    # ---------------------------------------------------------

    po = po.merge(
        supplier[
            [
                "supplier_id",
                "supplier_rating",
                "reliability_score",
            ]
        ],
        on="supplier_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Merge material information
    # ---------------------------------------------------------

    po = po.merge(
        material[
            [
                "material_id",
                "criticality",
                "lead_time_days",
            ]
        ],
        on="material_id",
        how="left",
        validate="many_to_one",
    )

    records = []

    for _, row in po.iterrows():

        reliability = float(
            row["reliability_score"]
        )

        rating = float(
            row["supplier_rating"]
        )

        criticality = row["criticality"]

        priority = row["priority"]

        # -----------------------------------------------------
        # Base delay probability
        # -----------------------------------------------------

        delay_probability = 0.10

        # Lower reliability -> higher delay risk
        delay_probability += (
            0.50
            * (1.0 - reliability)
        )

        # Lower supplier rating -> higher risk
        if rating < 3.0:
            delay_probability += 0.12

        elif rating < 4.0:
            delay_probability += 0.05

        # Material criticality
        if criticality == "CRITICAL":
            delay_probability += 0.08

        elif criticality == "HIGH":
            delay_probability += 0.04

        # Procurement priority
        if priority == "Critical":
            delay_probability += 0.04

        elif priority == "High":
            delay_probability += 0.02

        delay_probability = np.clip(
            delay_probability,
            0.02,
            0.90,
        )

        late_flag = int(
            rng.random()
            < delay_probability
        )

        # -----------------------------------------------------
        # Delay days
        # -----------------------------------------------------

        if late_flag:

            delay_days = int(
                rng.gamma(
                    shape=2.0,
                    scale=7.0
                    + (
                        12.0
                        * (1.0 - reliability)
                    ),
                )
            )

            delay_days = max(
                1,
                delay_days,
            )

        else:

            delay_days = int(
                rng.choice(
                    [0, 0, 0, 1, 2]
                )
            )

        # -----------------------------------------------------
        # Actual delivery date
        # -----------------------------------------------------

        actual_delivery_date = (
            row["expected_delivery_date"]
            + pd.Timedelta(
                days=delay_days
            )
        )

        # -----------------------------------------------------
        # Quantity variation
        # -----------------------------------------------------

        ordered_quantity = float(
            row["ordered_quantity"]
        )

        quantity_variation = rng.normal(
            loc=0,
            scale=0.03,
        )

        delivered_quantity = (
            ordered_quantity
            * (1 + quantity_variation)
        )

        delivered_quantity = max(
            0,
            delivered_quantity,
        )

        delivered_quantity = round(
            delivered_quantity,
            2,
        )

        # -----------------------------------------------------
        # Quality issue
        # -----------------------------------------------------

        quality_probability = 0.04

        quality_probability += (
            0.10
            * (1.0 - reliability)
        )

        quality_issue_flag = int(
            rng.random()
            < quality_probability
        )

        if quality_issue_flag:

            rejection_rate = rng.uniform(
                0.05,
                0.30,
            )

        else:

            rejection_rate = 0.0

        rejected_quantity = round(
            delivered_quantity
            * rejection_rate,
            2,
        )

        accepted_quantity = round(
            delivered_quantity
            - rejected_quantity,
            2,
        )

        # -----------------------------------------------------
        # Delivery status
        # -----------------------------------------------------

        if accepted_quantity <= 0:
            delivery_status = "Rejected"

        elif late_flag and quality_issue_flag:
            delivery_status = "Late - Quality Issue"

        elif late_flag:
            delivery_status = "Late"

        elif quality_issue_flag:
            delivery_status = "On Time - Quality Issue"

        else:
            delivery_status = "On Time"

        # -----------------------------------------------------
        # Record
        # -----------------------------------------------------

        records.append(
            {
                "delivery_id": (
                    f"DEL-{len(records) + 1:07d}"
                ),
                "po_id": row["po_id"],
                "project_id": row["project_id"],
                "supplier_id": row["supplier_id"],
                "material_id": row["material_id"],
                "planned_delivery_date": (
                    row["required_date"]
                ),
                "expected_delivery_date": (
                    row["expected_delivery_date"]
                ),
                "actual_delivery_date": (
                    actual_delivery_date
                ),
                "ordered_quantity": (
                    ordered_quantity
                ),
                "delivered_quantity": (
                    delivered_quantity
                ),
                "accepted_quantity": (
                    accepted_quantity
                ),
                "rejected_quantity": (
                    rejected_quantity
                ),
                "delay_days": delay_days,
                "late_delivery_flag": late_flag,
                "quality_issue_flag": (
                    quality_issue_flag
                ),
                "delivery_status": (
                    delivery_status
                ),
            }
        )

    return pd.DataFrame(records)