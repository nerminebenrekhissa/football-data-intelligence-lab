# Player Performance Prediction

This module focuses on predicting whether a football player will deliver a **good performance** in an upcoming match, using a **hybrid definition** that combines match ratings and statistical performance metrics.

It is designed as an explainable machine learning project.

---

## 1. Objectives

### Core Objective

- Predict if a player will have a **"good performance" in the next match**.

### Hybrid Label Definition

We will define "good performance" in two complementary ways:

1. **Rating-based label**
   - Use a match rating from a provider (probably flashscore).
   - Define a good performance as:  
     \- Rating ≥ a chosen threshold 
   - This creates a **binary classification** label:  
     \- `good_performance_rating = 1` if rating ≥ threshold, else `0`.

2. **Stats-based label**
   - Use performance metrics (xG, xA, shots, key passes, defensive actions, saves).
   - Build a composite or simple metric such as:
     - `offensive_contribution = xG + xA`
     - or a weighted index of several stats.
   - Define "good" as being above a certain percentile or threshold.
   - This creates a second binary label:  
     \- `good_performance_stats = 1` if above threshold, else `0`.

We will build and compare models using:
- **rating-based labels**
- **stats-based labels**

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
     - match ratings if available
   - Document the season(s), competition(s), and data source clearly in the module README and in `docs/`.

2. **Data Cleaning & Preparation**
   - Handle missing values, duplicates, and inconsistent records.
   - Ensure that players and matches are uniquely identifiable.
   - Create per-90 statistics where needed.
   - Filter by league, season, position, or minutes played to avoid noise.

3. **Feature Engineering**
   - Construct meaningful features such as:
     - rolling averages of key stats over last N matches
     - home vs away
     - opponent strength indicators
     - rest days between matches
     - position-specific features 

4. **Label Construction (Hybrid)**
   - Implement the rating-based label (good vs not good).
   - Implement the stats-based label (good vs not good).
   - Explore distributions and verify that the class balance is reasonable.

5. **Modeling**
   - Split data into train/validation/test sets.
   - Train baseline models:
     - Logistic Regression
     - Simple Decision Tree
   - Train more advanced models:
     - Random Forest
     - Gradient Boosting 
   - Compare performance between:
     - rating-based label model
     - stats-based label model

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
     - how recent form, opponent strength, or rest days affect predicted performance.

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
