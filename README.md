# Heart Disease Classification & Comparison

## Project Overview
This project implements and compares multiple machine learning models to predict the presence of heart disease based on clinical features. The objective is to perform binary classification (Disease vs. No Disease) and evaluate various algorithms to determine the best-performing model.

## Dataset
- **Source**: Cleveland Heart Disease Dataset
- **Observations**: 303 patients
- **Features**: 14 clinical attributes including:
  - Age, Sex, Chest Pain Type (cp)
  - Resting Blood Pressure (trestbps)
  - Cholesterol (chol)
  - Fasting Blood Sugar (fbs)
  - Resting ECG (restecg)
  - Maximum Heart Rate (thalach)
  - Exercise-Induced Angina (exang)
  - ST Depression (oldpeak)
  - Slope of ST Segment (slope)
  - Number of Major Vessels (ca)
  - Thalassemia (thal)
- **Target Variable**: Binary (0 = No Disease, 1 = Disease)

## Project Structure
```
ML Final Project/
├── dataset/
│   └── heart_disease.csv          # Raw dataset
├── report_images/                  # Generated plots for LaTeX report
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── cm_decision_tree.png
│   ├── cm_random_forest.png
│   ├── cm_logistic_regression.png
│   ├── cm_knn.png
│   ├── cm_svm.png
│   ├── cm_ann.png
│   └── model_comparison.png
├── Final_Project_Complete.ipynb   # Main notebook with all steps
├── report.tex                      # LaTeX project report
├── save_plots.py                   # Script to generate all plots
└── README.md                       # This file
```

## Methodology

### Phase I: Exploratory Data Analysis (EDA)
1. **Data Loading**: Read the Cleveland Heart Disease dataset
2. **Data Cleaning**:
   - Replaced missing values (marked as '?') with NaN
   - Converted all columns to numeric types
   - Imputed missing values using median (robust to outliers)
3. **Target Variable Analysis**:
   - Converted multi-class target (0-4) to binary (0 vs 1)
   - Analyzed class distribution
4. **Visualizations**:
   - Target distribution countplot
   - Correlation heatmap to identify feature relationships

### Phase II: Data Preprocessing
1. **Encoding**:
   - Applied One-Hot Encoding to categorical variables: `cp`, `restecg`, `slope`, `thal`
   - Used `drop_first=True` to avoid multicollinearity
2. **Train-Test Split**:
   - 80% training, 20% testing
   - Random state = 42 for reproducibility
3. **Feature Scaling** (for applicable models):
   - Applied `StandardScaler` (fit on train, transform on test)
   - Prevents data leakage
4. **Dimensionality Reduction**:
   - Applied PCA to retain 95% of variance
   - Reduced feature space while preserving information

### Phase III: Model Implementation & Evaluation

#### Models Requiring NO Scaling:
1. **Decision Tree Classifier**
   - Non-parametric, handles raw features well
   - Random state = 42

2. **Random Forest Classifier**
   - Ensemble of 100 decision trees
   - Random state = 42

#### Models Requiring Scaling + PCA:
3. **Logistic Regression**
   - Baseline linear model
   - Random state = 42

4. **K-Nearest Neighbors (KNN)**
   - k = 5 neighbors
   - Distance-based, sensitive to scale

5. **Support Vector Machine (SVM)**
   - Linear kernel
   - Random state = 42

6. **Artificial Neural Network (ANN)**
   - Multi-Layer Perceptron (MLPClassifier)
   - Architecture: 2 hidden layers (100, 50 neurons)
   - Activation: ReLU
   - Solver: Adam
   - Max iterations: 500
   - Random state = 42

### Phase IV: Comparison & Evaluation
- **Metrics Used**:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
- **Visualization**: Bar chart comparing all model accuracies

## How to Run

### Prerequisites
- Python 3.8+
- `uv` package manager (recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd "ML Final Project"

# Install dependencies using uv
uv add pandas numpy matplotlib seaborn scikit-learn
```

### Execution
1. **Run the main notebook**:
   ```bash
   jupyter notebook Final_Project_Complete.ipynb
   ```
   Or use your preferred notebook environment (VS Code, JupyterLab, etc.)

2. **Generate plots for report**:
   ```bash
   uv run python save_plots.py
   ```
   This will save all visualizations to `report_images/`

3. **Compile LaTeX report**:
   ```bash
   pdflatex report.tex
   bibtex report
   pdflatex report.tex
   pdflatex report.tex
   ```

## Results Summary
All models were evaluated on the same test set. The comparison plot (`model_comparison.png`) shows relative performance across all six algorithms.

**Key Findings**:
- Tree-based models (Decision Tree, Random Forest) performed well without scaling
- Scaled models (Logistic Regression, KNN, SVM, ANN) benefited from PCA
- Detailed results available in the notebook and LaTeX report

## Technologies Used
- **Python**: Core programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: Machine learning models and preprocessing
- **Jupyter Notebook**: Interactive development
- **LaTeX**: Professional report generation

## Project Requirements Met
✅ Exploratory Data Analysis (EDA)  
✅ Missing value handling  
✅ Statistical summary  
✅ Visualizations (histograms, correlation heatmap)  
✅ Data preprocessing (encoding, scaling, train-test split)  
✅ Implementation of 6 ML algorithms  
✅ PCA for dimensionality reduction (95% variance)  
✅ Model evaluation with confusion matrices  
✅ Comparison of all algorithms  
✅ Professional LaTeX report  

## Author
Hassan Jamshaid

## License
This project is for educational purposes as part of an ML course final project.
