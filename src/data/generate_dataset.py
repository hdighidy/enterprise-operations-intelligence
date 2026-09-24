from __future__ import annotations

from src.data.config import (
    NUM_EMPLOYEES,
    NUM_EQUIPMENT,
    NUM_MATERIALS,
    NUM_PROJECTS,
    NUM_SUPPLIERS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    create_directories,
)

from src.data.generators.employee_generator import (
    generate_employees,)

from src.data.generators.equipment_generator import (
    generate_equipment,)

from src.data.generators.material_generator import (
    generate_materials,)

from src.data.generators.supplier_generator import (
    generate_suppliers,)

from src.data.generators.project_generator import (
    generate_projects,)

from src.data.generators.activity_generator import (
    generate_project_activities,)

from src.data.generators.purchase_order_generator import (
    generate_purchase_orders,)


def main() -> None:

    print("=" * 60)
    print("Enterprise Operations Intelligence")
    print("Synthetic Data Generation")
    print("=" * 60)

    create_directories()

    print("\nGenerating materials...")
    materials = generate_materials(
        n=NUM_MATERIALS
    )

    print("Generating suppliers...")
    suppliers = generate_suppliers(
        n=NUM_SUPPLIERS
    )

    print("Generating employees...")
    employees = generate_employees(
        n=NUM_EMPLOYEES
    )

    print("Generating projects...")
    projects = generate_projects(
    n=NUM_PROJECTS,
    employee_ids=employees[
        "employee_id"
    ].tolist(),
    )

    print("Generating purchase orders...")
    purchase_orders = generate_purchase_orders(
        projects=projects,
        materials=materials,
        suppliers=suppliers,
        seed=RANDOM_SEED,
    )


    print("Generating project activities...")
    project_activities = (
       generate_project_activities(
           projects=projects,
           seed=42,
       )
    )

    print("Generating equipment...")
    equipment = generate_equipment(
        n=NUM_EQUIPMENT
    )

    materials.to_csv(
        RAW_DATA_DIR / "materials.csv",
        index=False,
    )

    suppliers.to_csv(
        RAW_DATA_DIR / "suppliers.csv",
        index=False,
    )

    employees.to_csv(
        RAW_DATA_DIR / "employees.csv",
        index=False,
    )

    projects.to_csv(
        RAW_DATA_DIR / "projects.csv",
        index=False,
    )

    purchase_orders.to_csv(
        RAW_DATA_DIR / "purchase_orders.csv",
        index=False,
    )

    project_activities.to_csv(
         RAW_DATA_DIR / "project_activities.csv",
        index=False,
    )

    equipment.to_csv(
        RAW_DATA_DIR / "equipment.csv",
        index=False,
    )

    print("\nGeneration completed.")

    print(f"Materials:  {len(materials):,}")
    print(f"Suppliers:  {len(suppliers):,}")
    print(f"Employees:  {len(employees):,}")
    print(f"Equipment:  {len(equipment):,}")
    print(f"Projects:   {len(projects):,}")
    print(f"Purchase Orders: {len(purchase_orders):,}")
    print(f"Activities: {len(project_activities):,}")
    print(f"\nFiles generated in:\n{RAW_DATA_DIR}")


if __name__ == "__main__":
    main()