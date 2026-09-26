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
from src.data.generators.delivery_generator import (
    generate_deliveries,)
from src.data.generators.material_consumption_generator import (
    generate_material_consumption,)
from src.data.generators.equipment_event_generator import (
    generate_equipment_events,)


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

    print("Generating equipment...")
    equipment = generate_equipment(n=NUM_EQUIPMENT)

    print("Generating purchase orders...")
    purchase_orders = generate_purchase_orders(
        projects=projects,
        materials=materials,
        suppliers=suppliers,
        seed=RANDOM_SEED,
    )

    print("Generating deliveries...")
    deliveries = generate_deliveries(
        purchase_orders=purchase_orders,
        suppliers=suppliers,
        materials=materials,
        seed=RANDOM_SEED,
    )


    print("Generating project activities...")
    project_activities = (
       generate_project_activities(
           projects=projects,
           seed=42,
       )
    )


    print("Generating material consumption...")
    material_consumption = generate_material_consumption(
        projects=projects,
        materials=materials,
        activities=project_activities,
        deliveries=deliveries,
        seed=42,
        max_activities=3000,
        materials_per_activity=2,
    )

    print("Generating equipment events...")
    equipment_events = generate_equipment_events(
        equipment=equipment,
        seed=42,
        max_equipment=500,
        days=180,
    )


    materials.to_csv(RAW_DATA_DIR / "materials.csv", index=False,)
    suppliers.to_csv(RAW_DATA_DIR / "suppliers.csv", index=False,)
    employees.to_csv(RAW_DATA_DIR / "employees.csv", index=False,)
    projects.to_csv(RAW_DATA_DIR / "projects.csv", index=False,)
    purchase_orders.to_csv(RAW_DATA_DIR / "purchase_orders.csv", index=False,)
    project_activities.to_csv(RAW_DATA_DIR / "project_activities.csv", index=False,)
    equipment.to_csv(RAW_DATA_DIR / "equipment.csv", index=False,)
    deliveries.to_csv(RAW_DATA_DIR / "deliveries.csv", index=False,)
    material_consumption.to_csv(RAW_DATA_DIR / "material_consumption.csv", index=False,)
    equipment_events.to_csv(RAW_DATA_DIR / "equipment_events.csv", index=False,)



    print("\nGeneration completed.")
    print(f"Materials:  {len(materials):,}")
    print(f"Suppliers:  {len(suppliers):,}")
    print(f"Employees:  {len(employees):,}")
    print(f"Equipment:  {len(equipment):,}")
    print(f"Projects:   {len(projects):,}")
    print(f"Purchase Orders: {len(purchase_orders):,}")
    print(f"Deliveries: {len(deliveries):,}")   
    print(f"Activities: {len(project_activities):,}")
    print(f"\nFiles generated in:\n{RAW_DATA_DIR}")
    print(f"Deliveries: {len(deliveries):,}")
    print(f"Material Consumption Records: {len(material_consumption):,}")
    print(f"Equipment Events: " f"{len(equipment_events):,}")



if __name__ == "__main__":
    main()