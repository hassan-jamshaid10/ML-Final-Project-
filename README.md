# Machine Learning Final Project: Heart Disease Prediction

Objective: Build and compare multiple machine learning models to predict whether a patient has heart disease based on clinical features.

## Project Structure

### 1. Setup & Data Loading
- **`Final_Project.ipynb`**: Initializes the project and downloads the dataset using `kagglehub`.
- **`dataset/`**: Contains the raw data (`heart_disease.csv`).

### 2. Exploratory Data Analysis (EDA)
- **`EDA.ipynb`**: 
  - Missing value analysis.
  - Statistical summary.
  - Visualizations (Distributions, Heatmaps).
  - Cleans the data and saves to `processed_data/`.

### 3. Data Preprocessing
- **`Data_Preprocessing.ipynb`**:
  - Target variable transformation (Binary classification).
  - One-Hot Encoding for categorical variables.
  - Train-Test Split (80/20).
  - Feature Scaling (StandardScaler) applied correctly to avoid leakage.
  - Saves processed train/test sets to `processed_data/`.

### 4. Model Implementation
Each model is implemented in its own notebook with **PCA (95% variance)** as a pre-processing step:
- **`Model_1_Logistic_Regression.ipynb`** (Baseline)
- **`Model_2_KNN.ipynb`**
- **`Model_3_SVM.ipynb`**
- **`Model_4_Decision_Tree.ipynb`**
- **`Model_5_Random_Forest.ipynb`**
- **`Model_6_ANN.ipynb`** (Artificial Neural Network)

## How to Run

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run Jupyter Notebook**:
   ```bash
   uv run jupyter notebook
   ```
   Or use the provided batch script:
   ```bash
   .\start_notebook.bat
   ```

3. **Execution Order**:
   Please run the notebooks in the following order to ensure data dependencies are met:
   1. `Final_Project.ipynb` (if you need to re-download data)
   2. `EDA.ipynb`
   3. `Data_Preprocessing.ipynb`
   4. Any `Model_X_....ipynb` file.

## Environment

This project uses `uv` for dependency management. Caches are stored locally in the `.cache` directory.
