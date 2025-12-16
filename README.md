# Statistical ML and OOD Generalization

This repository contains an end-to-end study of classical and neural machine learning models
for **regression, classification, feature importance analysis, out-of-distribution (OOD) evaluation,
transfer learning**, and **model deployment**.

The project is divided into four parts:
1. Regression with OOD generalization
2. Classification with feature selection
3. Transfer learning using pre-trained CNNs
4. Model deployment with an interactive web interface

---

## 1. Regression & Out-of-Distribution Prediction (Wine Quality)

### Dataset
- **Wine Quality Dataset** (Red & White wines)
- Source: UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/Wine+Quality

### Tasks
- Exploratory data analysis and preprocessing
- Feature scaling and cleaning
- Model training, validation, and testing
- Cross-domain OOD evaluation (Red → White, White → Red)

### Models
- Random Forest Regressor
- Support Vector Regression (RBF Kernel)
- Neural Network (Single Hidden Layer, Linear Output)

### Feature Importance
- Random Forest impurity-based importance
- Permutation importance
- Neural network sensitivity analysis
- Statistical validation using:
  - **ANOVA F-tests**
  - **Bayesian hypothesis testing**

### Key Analysis
- Compared feature importance consistency across model families
- Evaluated generalization under distribution shift (red vs white wines)

---

## 2. Classification: Down Syndrome Prediction in Mice

### Dataset
- **Mice Protein Expression Dataset**
- Source: UCI Machine Learning Repository  

### Objective
- Predict **genotype (binary classification)** using protein expression features
  from `DYRK1A_N` to `CaNA_N`.

### Preprocessing
- Exploratory analysis and visualization
- Missing value handling using **multivariate feature imputation**
  (IterativeImputer from scikit-learn)

### Models
- Random Forest Classifier
- Support Vector Classifier (RBF Kernel)
- Neural Network (Single Hidden Layer, Softmax Output)

### Feature Selection
- Recursive Feature Elimination with Cross-Validation (**RFECV**)
- Studied performance gains from systematic feature removal

### Evaluation Metrics
- Accuracy
- F1 Score

---

## 3. Transfer Learning & Feature Extraction

### Feature Extraction
- Used **ResNet-18** as a fixed feature extractor (PyTorch)
- Followed PyTorch transfer learning tutorial:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

### Implementation
- Implemented a function to extract **512-dimensional feature vectors**
  for each input image
- Generated Nx512 feature matrices for training and testing

### Models on Extracted Features
- RBF Kernel SVM
  - Grid search over kernel width and regularization
- Random Forest
  - Grid search over tree depth and number of estimators

### Evaluation
- Accuracy
- F1 Score

---

## 4. Model Deployment

- Deployed the wine quality regression model using **Streamlit**
- Interactive web UI with sliders for:
  - Acidity
  - Citric acid
  - Alcohol
  - Sulphates
  - Other chemical properties
- Demonstrated real-time predictions

---

## Advanced Feature Engineering

- Performed feature engineering and validation using:
  - Bayesian hypothesis testing
  - ANOVA F-tests
- Applied nonlinear dimensionality reduction:
  - **Kernel PCA**
  - **Sparse PCA**
- Improved representation quality and model robustness

---

## Tech Stack
- Python
- scikit-learn
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Streamlit
