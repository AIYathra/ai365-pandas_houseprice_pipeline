# House Prices Regression Pipeline

A clean, modular data pipeline using Pandas to transform the Kaggle House Prices dataset into an AI-ready, ML-ready format. This project includes data cleaning, exploratory data analysis (EDA), feature engineering, and baseline regression models, following best practices for Python data science projects.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Pipeline Components](#pipeline-components)
- [Usage Examples](#usage-examples)
- [Models](#models)
- [Baseline Results](#-baseline-results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete end-to-end pipeline for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The pipeline demonstrates best practices for:

- **Data Loading**: Efficient loading of CSV datasets with proper error handling
- **Data Cleaning**: Handling missing values, outliers, and data quality issues
- **Feature Engineering**: Creating meaningful features to improve model performance
- **Model Training**: Training and evaluating baseline regression models
- **Modular Design**: Reusable components that can be adapted for other projects

## 📁 Project Structure

```
ai365-pandas_houseprice_pipeline/
├── data/
│   ├── raw/                    # Raw data files (train.csv, test.csv)
│   │   └── .gitkeep
│   └── processed/              # Processed data and model outputs
│       └── .gitkeep
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and data analysis
│   └── 02_model_training.ipynb      # Model training and evaluation
├── src/
│   ├── __init__.py            # Package initialization
│   ├── load_data.py           # Data loading functions
│   ├── clean_data.py          # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation and transformation
│   └── train_model.py         # Model training and evaluation
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules

```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AIYathra/ai365-pandas_houseprice_pipeline.git
   cd ai365-pandas_houseprice_pipeline
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Quick Start

### 1. Download the Dataset

Download the Kaggle House Prices dataset from [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the `data/raw/` directory.

### 2. Run the Pipeline

**Option A: Using Python Scripts**

```bash
# Load and explore data
python src/load_data.py

# Clean the data
python src/clean_data.py

# Engineer features
python src/feature_engineering.py

# Train models
python src/train_model.py
```

**Option B: Using Jupyter Notebooks**

```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - notebooks/01_data_exploration.ipynb
# - notebooks/02_model_training.ipynb
```

## 📊 Dataset

The [House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) contains:

- **Training Set**: 1,460 houses with 79 features and sale prices
- **Test Set**: 1,459 houses with 79 features (no sale prices)
- **Features**: Mix of numerical and categorical variables describing various aspects of residential homes
- **Target**: `SalePrice` - the property's sale price in dollars

### Key Features

- **Living Area**: GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF
- **Quality Metrics**: OverallQual, OverallCond, ExterQual, KitchenQual
- **Age**: YearBuilt, YearRemodAdd
- **Location**: Neighborhood, MSZoning
- **Amenities**: Garage, Fireplace, Pool, etc.

## 🔧 Pipeline Components

### 1. Data Loading (`src/load_data.py`)

Functions to load training and test datasets:
- `load_train_data()`: Load training data
- `load_test_data()`: Load test data
- `load_data()`: Load both datasets

### 2. Data Cleaning (`src/clean_data.py`)

Data quality improvements:
- `handle_missing_values()`: Intelligent missing value imputation
- `remove_outliers()`: Z-score based outlier detection
- `get_data_quality_report()`: Data quality metrics and reporting

### 3. Feature Engineering (`src/feature_engineering.py`)

Creating new features:
- `create_total_area_features()`: Total square footage, bathrooms, porch area
- `create_age_features()`: House age, years since remodel, new construction flag
- `create_quality_features()`: Combined quality scores
- `encode_categorical_features()`: Label encoding for categorical variables

### 4. Model Training (`src/train_model.py`)

Training and evaluation:
- `train_linear_regression()`: Baseline linear model
- `train_ridge_regression()`: L2 regularized regression
- `train_random_forest()`: Ensemble tree-based model
- `evaluate_model()`: Calculate RMSE, MAE, R², RMSLE
- `cross_validate_model()`: K-fold cross-validation

## 💡 Usage Examples

### Example 1: Complete Pipeline

```python
from src.load_data import load_train_data
from src.clean_data import clean_data
from src.feature_engineering import engineer_features, select_features
from src.train_model import train_random_forest, evaluate_model, split_data

# Load data
train_df = load_train_data()

# Clean data
train_df = clean_data(train_df, is_training=True)

# Engineer features
train_df = engineer_features(train_df, encode=True)

# Prepare for modeling
X, y = select_features(train_df, 'SalePrice')
X_train, X_val, y_train, y_val = split_data(X, y)

# Train and evaluate
model = train_random_forest(X_train, y_train)
metrics = evaluate_model(model, X_val, y_val)
```

### Example 2: Data Quality Check

```python
from src.load_data import load_train_data
from src.clean_data import get_data_quality_report

# Load data
train_df = load_train_data()

# Get quality report
report = get_data_quality_report(train_df)
print(report[report['Missing_Count'] > 0])
```

## 🤖 Models

The pipeline includes three baseline models:

### 1. Linear Regression
- Simple baseline model
- Fast training and inference
- Good for understanding feature relationships

### 2. Ridge Regression
- L2 regularization to prevent overfitting
- Better generalization than linear regression
- Handles multicollinearity

### 3. Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Best performance among baseline models
- Feature importance analysis

### Evaluation Metrics

All models are evaluated using:
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (R-squared): Proportion of variance explained
- **RMSLE** (Root Mean Squared Log Error): Kaggle competition metric

## 📊 Baseline Results

Run `python src/train_model.py` (or `notebooks/02_model_training.ipynb`) to generate your baseline validation metrics and replace the placeholders below with your actual values.

| Model | RMSE ($) | MAE ($) | R² Score | RMSLE |
|-------|----------|---------|----------|-------|
| Linear Regression | `[fill from run]` | `[fill from run]` | `[fill from run]` | `[fill from run]` |
| Ridge Regression | `[fill from run]` | `[fill from run]` | `[fill from run]` | `[fill from run]` |
| Random Forest | `[fill from run]` | `[fill from run]` | `[fill from run]` | `[fill from run]` |

### How to interpret these metrics

- **RMSE**: Typical prediction error in dollars. Lower is better.
- **MAE**: Average absolute prediction error in dollars. Lower is better.
- **R² Score**: How much of the price variation the model explains (closer to 1.0 is better).
- **RMSLE**: Measures error on a log scale, so relative mistakes on cheaper and expensive homes are balanced (lower is better).

These baseline results are the performance **floor** for future improvements through feature engineering, tuning, and ensembling.

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning models and utilities
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Jupyter**: Interactive notebooks

## 📈 Future Enhancements

- Advanced feature engineering techniques
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Model stacking and ensembling
- Deep learning models
- Automated feature selection
- Pipeline automation with scikit-learn pipelines
- Model deployment with Flask/FastAPI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the House Prices dataset
- The Python data science community for excellent libraries and tools

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Modeling! 🏠📊**
