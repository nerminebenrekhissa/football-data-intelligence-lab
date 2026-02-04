
# Football Data Intelligence Lab

**Football Data Intelligence Lab** is an ongoing exploratory project focused on applying data analysis and machine learning concepts to real-world football datasets.

The project is designed as a modular “lab” environment to experiment with sports analytics methodologies commonly used in professional football analytics and data science research.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate football_lab
python player-similarity/src/predict_similarity.py
python player-performance/src/predict.py
```


## Project Overview

Football data is often fragmented, position-dependent, and difficult to compare fairly across players and teams.  
This project explores how machine learning and statistical analysis can be used to better understand player performance, player similarity, and match outcomes using real data from Europe’s Top-5 leagues.

---

## Project Modules

The repository is structured around three main modules:

### 1. Player Performance Prediction
Apply supervised learning models to player-level football statistics to predict player performance using season-level data as a proxy for form and quality.  
This module includes data preprocessing, feature engineering, normalization, and baseline machine learning models trained on real Top-5 league data.

### 2. Player Similarity & Scouting Assistant
Design and implement a player similarity analysis pipeline using engineered features, dimensionality reduction (PCA), and cosine similarity to identify players with similar statistical profiles.  
This module supports exploratory scouting-style analysis by enabling similarity search based on playing style rather than reputation or age.

### 3. Match Outcome Prediction
Explore match outcome prediction (Win / Draw / Loss) using team-level statistics, rolling averages, and recent form indicators derived from historical match data. *(planned)*

---

## Data

The project uses real football datasets covering Europe’s Top-5 leagues, including player-level statistics and advanced performance metrics.  
All preprocessing and feature engineering steps are documented through reproducible notebooks.

---

## Completed So Far

- Data collection and preprocessing (Top-5 European leagues)
- Dataset merging from multiple sources into a unified player-level table
- Feature engineering and normalization
- Design and implementation of a composite player performance score
- Baseline supervised machine learning models for player performance prediction
- Player similarity pipeline using PCA and cosine similarity


---

## Currently In Progress / Planned

- Deeper analysis and interpretation of model results
- Extension of player similarity with clustering and visualization
- Match outcome prediction using team-level statistics
- Optional Streamlit interface for interactive exploration

---

## Project Status

**This project is currently under active development.**  
The repository reflects an ongoing learning and research process rather than a finished production system.


---

## Notes

This project is exploratory in nature and is being continuously extended as part of ongoing learning in data science, machine learning, and sports analytics.




