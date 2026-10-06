# Model Explainability — Future Project Delay Prediction

## 1. Overview

This document explains the behavior of the predictive model developed for the **Enterprise Operations Intelligence** platform.

The objective is to identify project-day observations that are likely to experience a significant activity delay within the following 30 days.

### Prediction Objective

> **Given the operational state of a project today, can we identify whether a delay event will occur within the next 30 days?**

The model is designed as a decision-support capability rather than an autonomous decision maker.

It provides project and operational teams with:

- Early warning of potential delay
- Ranking of operational risk drivers
- Explainable reasons behind a risk prediction
- A foundation for proactive intervention

---

# 2. Model Selection Context

Four models were evaluated using the same chronological train/validation framework.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.5000 | 0.7735 | 0.7735 | 0.7735 | 1.0000 | 0.8723 |
| Logistic Regression | 0.9433 | **0.9837** | 0.8814 | 0.9662 | 0.8774 | 0.9196 |
| **Decision Tree** | **0.9521** | 0.9832 | **0.8878** | 0.9667 | **0.8854** | **0.9243** |
| Random Forest | 0.9407 | 0.9831 | 0.8738 | **0.9707** | 0.8628 | 0.9136 |

## Provisional Champion

The **Decision Tree** is currently selected as the champion validation model.

The selection is not based only on ROC-AUC.

The Decision Tree provides the best overall balance between:

- Predictive discrimination
- Precision
- Recall
- F1 score
- Interpretability
- Operational explainability

The improvement over Logistic Regression is relatively modest, therefore the result should be interpreted as a **strong but incremental improvement**, rather than a dramatic model advantage.

---

# 3. Validation Design

The evaluation follows a chronological split to prevent future observations from entering model training.

### Training

- Records: **92,909**
- Period: **2017-01-06 → 2026-06-25**
- Target rate: **65.83%**

### Validation

- Records: **11,026**
- Period: **2026-06-26 → 2028-07-05**
- Target rate: **77.35%**

### Test

- Records: **2,211**
- Period: **2028-07-06 → 2030-07-17**
- Status: **LOCKED**

The test set is intentionally excluded from model selection and explainability decisions.

---

# 4. Explainability Strategy

The project uses two complementary explainability approaches.

## 4.1 Global Explainability

Global explainability answers:

> Which operational variables are most influential across the model?

For the Decision Tree, the primary method is:

- Tree feature importance
- Feature ranking
- Aggregated business interpretation

For Logistic Regression, coefficient magnitude and direction provide an additional interpretable baseline.

---

## 4.2 Local Explainability

Local explainability answers:

> Why was this particular project-day classified as high risk?

The intended production workflow is:

```text
Project-Day
     │
     ▼
Feature Vector
     │
     ▼
Champion Model
     │
     ├── Risk Probability
     │
     ├── Risk Classification
     │
     └── Explanation
            │
            ├── Current delay condition
            ├── Recent delay trend
            ├── Material availability
            ├── Equipment disruption
            └── Cost exposure
```

This transforms a model prediction into an operationally understandable risk signal.

---

# 5. Current Explainability Findings

The earlier Logistic Regression analysis provides an interpretable reference point for understanding the model behavior.

The strongest positive coefficients included:

| Feature | Direction |
|---|---|
| `activity_delay_days` | Positive |
| `project_complexity_VERY_HIGH` | Positive |
| `project_complexity_HIGH` | Positive |
| `equipment_failure_count_rolling_30d` | Positive |
| `delayed_activity_count` | Positive |
| `activity_delay_days_rolling_14d` | Positive |
| `activity_delay_days_lag_1` | Positive |
| `activity_delay_days_lag_7` | Positive |
| `material_stockout_count_rolling_30d` | Positive |
| `active_activity_count` | Positive |
| `elapsed_days` | Positive |
| `delay_signal_flag` | Positive |

The strongest negative coefficients included:

| Feature | Direction |
|---|---|
| `equipment_downtime_hours_rolling_30d` | Negative |
| `planned_duration_days` | Negative |
| `planned_progress_pct` | Negative |
| `active_activity_count_lag_7` | Negative |
| `activity_delay_days_rolling_7d` | Negative |
| `material_stockout_count_rolling_14d` | Negative |
| `activity_delay_days_rolling_30d` | Negative |
| `delayed_activity_count_lag_7` | Negative |
| `delayed_activity_count_lag_1` | Negative |

These coefficients describe model association, **not causal relationships**.

---

# 6. Key Risk Driver: Current Activity Delay

The most important observed signal is:

```text
activity_delay_days
```

The relationship between current activity delay and the future target is particularly strong in the synthetic dataset.

| Current Activity Delay | Future Delay Target Rate |
|---|---:|
| ≤ 0 days | 28.31% |
| 1–3 days | 37.10% |
| 3–7 days | 98.52% |
| 7–14 days | 98.68% |
| 14–30 days | 99.22% |
| 30+ days | 99.75% |

This feature is **not classified as target leakage** because it represents the observed operational state at the prediction date.

However, its unusually strong relationship with the future target is treated as a **synthetic-data dependency**.

Therefore:

> The model demonstrates strong predictive behavior on the generated dataset, but the magnitude of this relationship must be validated against real enterprise project data before production deployment.

---

# 7. Operational Risk Score

`operational_risk_score` is a composite operational indicator.

Its components are:

| Component | Weight |
|---|---:|
| Delayed activity ratio | 30% |
| Material stockout ratio | 30% |
| Equipment downtime | 20% |
| Cost variance | 20% |

The feature is calculated from current operational observations and does not directly use the final project delay target.

Therefore it is considered **prediction-time valid**.

It should nevertheless be interpreted as a derived operational indicator rather than an independent causal variable.

---

# 8. Delay Signal Flag

`delay_signal_flag` is a binary operational deterioration indicator.

It becomes active when at least one of the following conditions is present:

```text
Delayed activities > 0
OR
Material stockouts > 0
OR
Cost variance > 10%
```

This feature is useful from an operational perspective because it converts several operational symptoms into a simple early-warning signal.

It is considered prediction-time valid because it is constructed from current operational observations rather than the final project outcome.

---

# 9. Prediction-Time Feature Contract

The model follows a strict distinction between information available at prediction time and information that becomes known only after the project outcome.

### Allowed Predictors

```text
contract_value
planned_duration_days
project_complexity

elapsed_days
planned_progress_pct

active_activity_count
critical_activity_count
delayed_activity_count
activity_delay_days

material_stockout_count
material_availability_ratio

delivery_count
delivered_quantity

equipment_downtime_hours
equipment_failure_count
equipment_maintenance_cost

cost_variance
cost_variance_pct

operational_risk_score
delay_signal_flag

historical lag features
rolling historical features
```

### Forbidden Predictors

```text
project_delay_target
delay_flag
delay_days
actual_finish_date
actual_duration_days
final_delay_days
delay_next_30_days
```

The final target is deliberately excluded from the predictor set.

---

# 10. Why the Model Is Explainable

The Decision Tree provides a natural rule-based representation of operational risk.

Conceptually:

```text
                 Project-Day
                     │
          ┌──────────┴──────────┐
          │                     │
   Current Delay?          Operational State?
          │                     │
     ┌────┴────┐          ┌─────┴─────┐
     │         │          │           │
   Low       High      Materials   Cost/Equipment
     │         │          │           │
     ▼         ▼          ▼           ▼
 Lower       Higher     Higher      Higher
 Risk         Risk       Risk        Risk
```

The actual tree rules will be extracted programmatically from the fitted validation model rather than manually invented.

This is important because the explanation should always reflect the actual trained model.

---

# 11. Business Interpretation

The model should not simply communicate:

> "Project = High Risk"

Instead, the operational intelligence layer should communicate:

> **High delay risk detected**

with supporting drivers such as:

```text
Primary driver:
Current activity delay

Supporting signals:
• Recent activity delays increasing
• Material stockout activity detected
• Elevated operational risk score
• Recent equipment disruption
• Cost variance above normal level
```

This allows the model to support an intervention workflow:

```text
Detect
   ↓
Explain
   ↓
Prioritize
   ↓
Investigate
   ↓
Act
   ↓
Monitor
```

---

# 12. Important Modeling Limitation

The current dataset is synthetic.

Therefore:

- Feature relationships are generated rather than observed from real projects.
- Some relationships may be stronger than they would be in production.
- The target-generation mechanism can influence model performance.
- Equipment signals are currently enterprise-level because equipment events do not yet contain `project_id`.
- The model should not be presented as having demonstrated real-world production accuracy.

The current result should therefore be described as:

> **A controlled synthetic benchmark demonstrating an end-to-end explainable predictive-risk workflow.**

---

# 13. Explainability Governance

The project follows these principles:

### No causal claims

Feature importance indicates model association, not causality.

### No hidden target features

Final project outcomes are excluded from predictors.

### Temporal integrity

Historical lag and rolling features are constructed using information available before the prediction date.

### Locked test set

The test set is not used for feature selection or model tuning.

### Human decision support

Predictions are intended to support project and operational teams rather than automatically execute business decisions.

---

# 14. Reproducibility

The explainability workflow is generated from the same model pipeline used for validation.

Example:

```powershell
python -m src.models.tree_models
```

Then the explainability analysis can be generated from the fitted model.

The resulting artifacts should be stored under:

```text
reports/
├── model_comparison.csv
├── decision_tree_feature_importance.csv
├── decision_tree_rules.txt
└── model_explainability.md
```

---

# 15. Next Explainability Outputs

The next implementation stage will generate the actual Decision Tree artifacts:

```text
Decision Tree
      │
      ├── Feature Importance
      │
      ├── Top 20 Features
      │
      ├── Tree Rules
      │
      ├── Risk Driver Ranking
      │
      └── Business Interpretation
```

These outputs will be generated directly from the trained model so that the documentation remains synchronized with the actual model.

---

# 16. Executive Summary

### Current Champion

**Decision Tree**

### Validation Performance

- ROC-AUC: **0.9521**
- PR-AUC: **0.9832**
- Precision: **0.9667**
- Recall: **0.8854**
- F1: **0.9243**

### Primary Observed Risk Signal

**Current activity delay**

### Other important operational signals

- Project complexity
- Recent activity delays
- Material stockouts
- Equipment failures/downtime
- Cost variance
- Operational risk score
- Project schedule progression

### Model Status

**Provisional champion — temporal validation complete**

### Production Status

**Not production validated**

The model currently demonstrates strong performance on the synthetic enterprise dataset and provides an interpretable foundation for proactive project-delay risk management. Real-world validation remains necessary before operational deployment.