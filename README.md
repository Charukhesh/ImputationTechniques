# Impact of Regression-Based Imputation on Credit Default Prediction

## 📖 Project Overview

This project tackles a common and critical challenge in machine learning: handling missing data. We explore this challenge within the context of a credit risk assessment task, using the UCI Credit Card Default Clients dataset. The core idea is that the method chosen to handle missing values can significantly impact the performance of a final predictive model.

To create a realistic testbed, we begin by **artificially introducing missing values** into several key numerical columns. From there, we implement and compare four distinct strategies:

1.  **Listwise Deletion:** The naive approach of simply discarding incomplete records.
2.  **Median Imputation:** A simple statistical baseline where missing values are replaced with the column's median.
3.  **Linear Regression Imputation:** A model-based approach that predicts missing values assuming a linear relationship with other features.
4.  **K-Nearest Neighbors (KNN) Imputation:** A more sophisticated, non-linear model that uses the 'k' most similar data points to predict missing values.

The effectiveness of these strategies is not judged by the accuracy of the imputed values themselves, but by the performance of a **downstream Logistic Regression classifier** trained on each of the resulting datasets. By comparing classification metrics like the **F1-Score**, we can directly measure which data preparation technique ultimately leads to a more reliable credit default prediction model.

## 📁 Repository Structure

```
ImputationTechniques/
│
├── main.ipynb
├── creditcard_dataset/
│   └── UCI_Credit_Card.csv
└── README.md
```

- **main.ipynb**: The main Jupyter notebook containing all code, analysis, visualizations, and conclusions.
- **creditcard_dataset/**: Directory containing the dataset used for analysis.
    - **UCI_Credit_Card.csv**: The Credit Card Default Clients dataset from Kaggle.
- **README.md**: Project documentation and instructions.

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run This Project
1. **Clone the Repository**
    ```bash
    git clone https://github.com/Charukhesh/ImputationTechniques.git
    cd ImputationTechniques
    ```

2. **Dataset**

    Ensure the UCI_Credit_Card.csv file is present in the creditcard_dataset/ folder.

3. **Open and Run the Notebook**
    - Open `main.ipynb` in [Jupyter Notebook](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/).
    - Run all cells sequentially to reproduce the analysis and results.
