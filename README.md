[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive machine learning solution for detecting fraudulent credit card transactions using advanced AI techniques and model explainability.

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for credit card fraud detection, showcasing skills in data preprocessing, model selection, evaluation, and deployment. The solution handles class imbalance, provides model interpretability, and includes a user-friendly web interface.

### Key Features
- 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, Neural Networks
- ⚖️ **Class Imbalance Handling**: SMOTE oversampling techniques
- 📊 **Model Explainability**: SHAP and LIME integration
- 🌐 **Interactive Dashboard**: Streamlit web application
- 🔍 **Anomaly Detection**: Autoencoder implementation
- 📈 **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📊 Dataset Information

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (28 PCA-transformed + Time + Amount)
- **Target**: Binary classification (Fraud: 0.17%, Normal: 99.83%)
- **Challenge**: Highly imbalanced dataset

## 🏗️ Project Structure

```
AI Credit card Project/
├── data/
│   └── creditcard.csv            # Credit card fraud dataset
├── app.py                        # Main Streamlit application
├── dashboard.py                  # Dashboard components
├── preprocessing.py              # Data preprocessing utilities
├── logistic_regression.py        # Logistic Regression model
├── random_forest.py              # Random Forest model
├── xgboost_model.py              # XGBoost model
├── neural_network.py             # Neural Network model
├── evaluate_model.py             # Model evaluation and metrics
├── main.py                       # Main execution script
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-credit-card-fraud-detection.git
cd ai-credit-card-fraud-detection
```

### 2. Set Up Environment
```bash
# Using pip
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Follow instructions in data/README.md to download from Kaggle
# Place creditcard.csv in the data/ directory
```

### 4. Run the Application
```bash
# Run the Streamlit dashboard
streamlit run app.py

# Or run the main script
python main.py
```

## 🔬 Methodology

### Data Preprocessing
- **Feature Scaling**: StandardScaler for numerical features
- **Class Imbalance**: SMOTE (Synthetic Minority Oversampling Technique)
- **Data Splitting**: Stratified train/test split (80/20)

### Models Implemented

| Model | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.92 | 0.88 | 0.94 |
| Random Forest | 0.96 | 0.75 | 0.84 | 0.95 |
| **XGBoost** | **0.94** | **0.89** | **0.91** | **0.97** |
| Neural Network | 0.90 | 0.79 | 0.84 | 0.93 |

*XGBoost achieved the best overall performance*

### Evaluation Metrics
- **Confusion Matrix**: Visual representation of predictions
- **Precision**: Minimizing false positives (important for fraud detection)
- **Recall**: Maximizing true positives (catching actual fraud)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 📈 Key Results

### Model Performance Highlights
- **Best Model**: XGBoost with 97% AUC-ROC
- **High Precision**: 96% precision with Random Forest (minimal false alarms)
- **High Recall**: 92% recall with Logistic Regression (catches most fraud)
- **Balanced Performance**: XGBoost provides optimal precision-recall trade-off

### Business Impact
- **Cost Reduction**: Minimized false positives reduce unnecessary card blocks
- **Fraud Prevention**: High recall ensures maximum fraud detection
- **Real-time Scoring**: Models optimized for low-latency predictions

## 🔍 Model Explainability

### SHAP (SHapley Additive exPlanations)
- Global feature importance analysis
- Local prediction explanations
- Feature interaction effects

### LIME (Local Interpretable Model-agnostic Explanations)
- Individual prediction explanations
- Feature contribution visualization
- Model-agnostic approach

## 🌐 Web Application

### Streamlit Dashboard
- Interactive fraud prediction interface
- Real-time model performance metrics
- Feature importance visualizations
- Batch prediction capabilities

## 🔧 Advanced Features

### Anomaly Detection
- **Autoencoder Neural Networks**: Unsupervised fraud detection
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector machine approach

### Model Monitoring
- **Data Drift Detection**: Monitor feature distribution changes
- **Model Performance Tracking**: Continuous evaluation metrics
- **Alert System**: Automated notifications for model degradation

## 📊 Visualizations

The project includes comprehensive visualizations:
- Distribution plots for fraud vs normal transactions
- Correlation heatmaps
- ROC curves comparison
- Precision-recall curves
- Feature importance plots
- SHAP summary and waterfall plots

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Neural network implementation
- **SHAP/LIME**: Model explainability
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Imbalanced-learn**: SMOTE implementation

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning Pipeline**: End-to-end ML workflow
- **Class Imbalance Handling**: Techniques for skewed datasets
- **Model Selection**: Comparing multiple algorithms
- **Evaluation Metrics**: Understanding business-relevant metrics
- **Model Interpretability**: Explaining AI decisions
- **Web Development**: Creating interactive applications
- **Version Control**: Git best practices
- **Documentation**: Professional project presentation

## 🚀 Deployment Options

### Local Deployment
```bash
# Run Streamlit app
streamlit run app.py

# Run main script
python main.py
```

### Cloud Deployment
- **Heroku**: Easy web app deployment
- **AWS SageMaker**: Scalable ML model serving
- **Google Cloud Run**: Containerized application deployment
- **Azure ML**: End-to-end ML lifecycle management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Inspired by real-world fraud detection challenges
- Built for educational and portfolio purposes

## 📞 Contact

- **GitHub**: https://github.com/Kalpan25
- **LinkedIn**: [Kalpan Patel](https://www.linkedin.com/in/kalpanpatel30/)
- **Email**: kalpan.p.patel30@gmail.com

---

⭐ **Star this repository if you found it helpful!** ⭐

