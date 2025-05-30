# Premier League Match Predictor

## Introduction

This repository contains a machine learning project aimed at predicting the number of goals scored in Premier League matches. Various models were explored, including LightGBM, MLP, KNN, Random Forest, and XGBoost.

Through experimentation and evaluation, **LightGBM** demonstrated the best performance on our data. To simplify usage and ensure clarity, a single folder — `Final Model` — Includes all the final and best files: the best modifyed pipeline, and the LightGBM notebook where the model evaluation is shown.

For comparison, we also include a complete notebook for the second-best performing model, the MLP (Multi-Layer Perceptron). The remaining models (KNN, Random Forest, XGBoost, etc.) are accessible via a modular pipeline that allows each team member to experiment independently with consistent data transformations.

---

## Usage

Follow the steps below to set up and run the LightGBM model:

### 1. Create and activate a virtual environment and Install required dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Update the database to include the latest Premier League matches

Run the following script to fetch the most up-to-date match data:

```bash
python3 scrape_epl.py
```
This script downloads the latest data and updates the SQLite database used in the notebooks. Note that for this reason, if the notebook is runned after a 1 week period, the results inlcuding performance metrics, graphs ecc., will be different due to new data (new Premier League matches) coming in.

---

### 3. Run the LightGBM pipeline

Navigate to and run all cells in `Final_LightGBM_Model.ipynb`, inside the `Final Model` folder. This notebook contains:


- LightGBM training and evaluation

- Final results and model interpretation

---

### Additional Model

For how the EDA and Feature Engineering was initially computed, go to `ML_project.ipynb`
As stated before, we also provide a full notebook implementation for MLP, the second-best performing model. To run it, follow the same steps as above for the Jupyter notebook `MLP_notebook.ipynb`

---

### Other Models (KNN, Random Forest, XGBoost, etc.)

For experimentation with additional models, we implemented a shared pipeline structure. This ensures:

- Consistent preprocessing

- Reproducibility across experiments

- Flexibility for individual team members to apply model-specific tuning

Each model has its own notebook under the `notebooks/` directory, where it can be tested independently using the corresponding modyfied pipeline that includes the changes made to the feature engineering made to improve that model.


