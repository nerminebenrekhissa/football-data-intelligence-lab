# Match Outcome Prediction

This module focuses on predicting the **result of a football match** and the **probability of match outcomes**, using team-level statistics, form indicators, and expected goals (xG).

---

## 1. Objectives

### Core Objective – Classification (Win/Draw/Loss)

- Predict the outcome of a match as one of three classes:
  - Home Win
  - Draw
  - Away Win

This is a **multi-class classification** problem.

### Advanced Objective – Probabilistic Modeling

- Predict the **probabilities** of each outcome:
  - P(Home Win), P(Draw), P(Away Win)
- Evaluate not only classification accuracy, but also the **quality of probability estimates** through:
  - calibration curves
  - log loss / cross-entropy
  - Brier score 

### xG Integration (Expected Goals)

As a modern extension, we will integrate **expected goals (xG)**:

- Use xG and xGA (expected goals against) as key features in the model.
- Optionally:
  - build a secondary model to predict xG for a team based on pre-match features.
  - analyze how xG-driven features affect the predicted outcome probabilities.

---

## 2. Planned Workflow

1. **Data Collection**
   - Acquire real match-level datasets that include:
     - team names
     - match date
     - final score
     - basic stats (shots, possession, etc.)
     - ideally xG data 
   - Cover at least one full season of a major league (or multiple leagues).
   - Document sources and seasons clearly.

2. **Data Preparation & Feature Engineering**
   - Create features such as:
     - home vs away indicator
     - goal difference trends
     - points per game in recent matches
     - form indicators (last N results)
     - rolling averages of stats 
   - If xG is available:
     - compute rolling xG for attack and defense.
   - Build target variable:
     - outcome: Home Win / Draw / Away Win.

3. **Baseline Modeling**
   - Simple baseline models:
     - majority class
     - naive home advantage model
   - First ML models:
     - Logistic Regression 
     - Random Forest / Gradient Boosting

4. **Probability Modeling**
   - Use models that output probabilities:
     - Logistic Regression
     - Gradient Boosting / XGBoost / LightGBM
   - Evaluate:
     - accuracy and macro-F1
     - confusion matrix
     - log loss / cross-entropy
     - calibration (how good the predicted probabilities are)

5. **xG-Enhanced Models**
   - Add xG-related features:
     - season xG metrics
     - rolling xG for attack and defense
   - Compare performance:
     - with vs without xG
   - Optionally:
     - build a model that predicts expected goals, then use it to inform outcome prediction.

6. **Visualization & Reporting**
   - Visualize:
     - feature importance
     - relationships between rolling form and win probability
     - calibration plots
   - Provide a written summary of:
     - which features are most predictive
     - how well the model discriminates between strong and weak teams
     - limitations 

---

## 3. Module Structure

```text
match-outcome/
├─ README.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
├─ models/
└─ reports/
   └─ figures/
