
# Player Similarity & Scouting Assistant

This module builds a **player similarity and scouting assistant**, capable of identifying players who are statistically similar to a given reference player.

It uses a combination of feature engineering, dimensionality reduction, and similarity metrics.

---

## 1. Objectives

### Core Objective

- Represent players in a **feature space** based on their per-90 or season-level statistics.
- Allow querying of:
  - “Top N most similar players to player X” using cosine similarity.

### Advanced Objectives

- Extend **dimensionality reduction** to:
  - enable 2D visualization of the player space.
- Apply **clustering** to:
  - group players into roles or profiles.
- Create **visual scouting tools**:
  - PCA 2D scatter plots with clusters
  - radar charts comparing two players
  - cluster “profiles” or archetypes

---

## 2.Workflow

1. **Data Collection**
   - Obtain a dataset of player statistics for at least one league/season (or multiple).
   - Use per-90 or rate-based metrics where possible.
   - Include different types of metrics:
     - passing
     - shooting
     - defensive actions
     - progression
     - possession involvement

2. **Feature Selection & Preprocessing**
   - Select a subset of relevant features (avoid redundant or overly correlated stats if needed).
   - Handle missing values.
   - Apply scaling/standardization so all features are on comparable scales.

3. **Dimensionality Reduction**
   - Apply PCA (or similar) to:
     - reduce dimensionality for visualization
     - possibly denoise the features.
   - Inspect explained variance and choose an appropriate number of components.

4. **Clustering (Future Extension)**
   - Apply clustering on:
     - standardized features
     - PCA components.
   - Explore different numbers of clusters.
   - Interpret clusters as rough “roles” or “profiles”.

5. **Similarity Search**
   - Use a similarity metric to:
     - find the most similar players to a target player.
   - Implement:
     - a function that returns top N similar players.
     - optional filters (by league, position, minutes played, age).

6. **Visualization & Scouting Use Cases (Future Extension)**
   - Visualizations:
     - 2D PCA plot with color-coded clusters.
     - Radar charts comparing:
       - target player vs similar players
       - cluster centroids vs individual players.
   - Scouting scenarios:
     - For example: “Find young midfielders in league X who are statistically similar to Pedri.”
   - Provide written analysis explaining:
     - how the similarity method works
     - limitations.

---

## 3. Module Structure

```text
player-similarity/
├─ README.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
├─ models/
└─ reports/
   └─ figures/
