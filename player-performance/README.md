# Player Performance Prediction

This module focuses on predicting whether a football player will deliver a **good performance** in an upcoming match, using season-level performance statistics as a proxy for player form and quality.

It is designed as an explainable machine learning project.

Models are trained and evaluated using the statistics-based proxy performance label derived from the composite performance_score.


---

## 1. Objectives

### Core Objective

- Predict whether a player is likely to deliver a good performance in an upcoming match, based on learned patterns from historical season-level statistics.

### Performance Label Definition

In the current version of the project, "good performance" is defined using a statistics-based proxy label.

Use a statistics-based proxy label built from a composite performance_score derived from multiple performance dimensions, including:
- attacking contribution
- defensive contribution
- possession involvement
- discipline
- minutes played

The performance_score is computed using position-specific weights to account for different player roles.
- A player is labeled as "good performance" if their performance_score is above a defined percentile threshold, resulting in a binary classification label.

Because match-by-match performance labels are not available in the current dataset, season-level composite performance is used as a proxy target to learn patterns associated with strong player performance, as a first step toward future match-level prediction.


### Advanced Objective (Extension – Injury Risk)

As an optional extension (if data allows), we will explore a basic **injury risk prediction** task, for example:

- Predict whether a player will **suffer an injury or miss matches** in the near future, using:
  - workload (minutes played, games in short periods)
  - age
  - position
  - physical metrics if available
  - historical injury patterns

This extension will be implemented only if suitable data can be collected and will be clearly separated as an advanced part of the module.

---

## 2. Planned Workflow

1. **Data Collection**
   - Identify and download a real dataset with:
     - player-level match stats by game
     - per-90 or per-match statistics
     - match ratings if available (future extension)
   - Document the season(s), competition(s), and data source clearly in the module README and in `docs/`.

2. **Data Cleaning & Preparation**
   - Handle missing values, duplicates, and inconsistent records.
   - Ensure that players and matches are uniquely identifiable.
   - Create per-90 statistics where needed.
   - Filter by league, season, position, or minutes played to avoid noise.

3. **Feature Engineering**
   - Construct meaningful features such as:
     - per-90 and aggregated season-level statistics
     - position-specific performance indicators
     - composite performance metrics
 

4. **Label Construction (Current & Future)**
  - Construct a statistics-based proxy label using the composite performance_score.
  - Explore the distribution of the label and verify class balance.
  - (Future extension) Integrate rating-based labels from external providers such as Flashscore to enable true match-level performance prediction.


5. **Modeling**
   - Split data into train/validation/test sets.
   - Train baseline models:
     - Logistic Regression
     - Simple Decision Tree
   - Train more advanced models:
     - Random Forest
     - Gradient Boosting 
   - Evaluate model performance using the statistics-based proxy label.
   - (Future extension) Compare performance against rating-based labels once such data is integrated.


6. **Evaluation**
   - Use appropriate metrics:
     - Accuracy
     - Precision, Recall, F1-score (especially for the "good" class)
     - ROC-AUC (if applicable)
   - Analyze performance per:
     - player position
     - team
     - competition / league

7. **Interpretability & Insights**
   - Use feature importance or SHAP (if feasible) to explain:
     - which features contribute most to predictions
   - Provide football-context interpretation:
     - how different statistical performance dimensions and player roles influence predicted performance.


8. **(Optional) Injury Risk Extension**
   - If an injury dataset is obtained:
     - define an injury label which means missing matches due to injury in upcoming period
     - engineer workload features
     - build and evaluate a simple classifier
     - Clearly document limitations and assumptions.

---

## 3. Module Structure

```text
player-performance/
├─ README.md
├─ data/
│  ├─ raw/          # original data (read-only)
│  └─ processed/    # cleaned data
├─ notebooks/       # EDA and experiments
├─ src/             # reusable Python code
├─ models/          # saved models
└─ reports/
   └─ figures/      # plots and visualizations
