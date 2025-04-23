# EPL xG Prediction

## Project Overview

This project implements a machine learning pipeline to predict Expected Goals (xG) for English Premier League (EPL) teams on a per-match basis using seasonal data. The goal is to quantify and forecast offensive and defensive performance, helping analysts and fans better understand team strengths and upcoming match expectations.

### Key Components

- **Data Collection & Preprocessing**: Pulls historical EPL data from sources like Opta or StatsBomb, cleans it, normalizes formats, and creates initial variables such as shots, possession, team ratings, and form indicators.
- **Feature Engineering**: Builds advanced features including moving averages, shot quality metrics, home/away splits, and situational factors like rest days or injuries.
- **Modeling**: Trains regression and machine learning models (e.g., Poisson regression, XGBoost, Random Forest) and neural networks to predict xG.
- **Evaluation**: Assesses model performance with RMSE, MAE, cross-validation, and visual tools like calibration plots.
- **Deployment**: Provides scripts for making predictions and Jupyter notebooks for analysis.

## Repository Structure

```
├── data/                   # Datasets (raw and processed)
│   ├── raw/                # Original data downloads
│   └── processed/          # Cleaned and feature-enhanced datasets
├── src/                    # Source code
│   ├── data_pipeline/      # Data ingestion and preprocessing
│   ├── features/           # Feature engineering scripts
│   ├── models/             # Model training and prediction
│   ├── evaluation/         # Evaluation tools and metrics
│   └── utils/              # General helper functions
├── notebooks/              # Interactive notebooks for analysis
├── outputs/                # Output files (e.g., trained models, plots)
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/epl-xg-prediction.git
   cd epl-xg-prediction
   ```

2. **Set up the environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the Pipeline

1. **Download and preprocess data**:
   ```bash
   python src/data_pipeline/fetch_data.py --season 2023-2024
   python src/data_pipeline/process_data.py
   ```

2. **Create features**:
   ```bash
   python src/features/build_features.py
   ```

3. **Train the model**:
   ```bash
   python src/models/train_model.py --model xgboost
   ```

4. **Evaluate performance**:
   ```bash
   python src/evaluation/evaluate_model.py --model xgboost
   ```

5. **Make predictions**:
   ```bash
   python src/models/predict.py --fixture_list fixtures.csv
   ```

## Git Workflow

This project uses a **Feature Branch Workflow** with Git to maintain modular development and clean integration:

1. **Main Branch (`main`)**:
   - Holds the production-ready version of the project.
   - Protected from direct commits; changes must go through Pull Requests (PRs).

2. **Develop Branch (`develop`)** *(optional)*:
   - A staging branch that combines features before merging into `main`.

3. **Feature Branches**:
   - Created off `develop` (or `main` if no `develop` branch exists).
   - Naming convention: `feature/<short-description>` (e.g., `feature/xg-model`)
   - Example:
     ```bash
     git checkout develop
     git pull origin develop
     git checkout -b feature/add-xg-model
     # make changes
     git add .
     git commit -m "Add xG prediction model"
     git push origin feature/add-xg-model
     ```

4. **Pull Requests**:
   - Open a PR into `develop` (or `main`).
   - Include a summary of changes, relevant screenshots, or test results.
   - Reviewers provide feedback. Resolve comments before merging.
   - Pass all tests (`pytest`) and follow linting guidelines (e.g., `flake8`).

5. **Merging**:
   - Once approved, squash or merge the PR.
   - Delete the feature branch after merging.
   - CI/CD pipelines will run post-merge.

6. **Hotfix Branches**:
   - For critical fixes, create from `main`: `hotfix/<issue-name>`.
   - Merge back into both `main` and `develop`.

## Contributing

Interested in improving this project? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) guide to learn how to contribute, report bugs, or suggest features.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Maintainers

- **Lead Developer**: Jane Doe – [jane.doe@example.com](mailto:jane.doe@example.com)
- **Data Engineer**: John Smith – [john.smith@example.com](mailto:john.smith@example.com)