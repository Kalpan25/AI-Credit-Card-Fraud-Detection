# 🔒 AI-Based Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A machine learning project for detecting fraudulent credit card transactions using multiple algorithms and a simple web interface.

## 🎯 Project Overview

This project demonstrates a basic machine learning pipeline for credit card fraud detection, showcasing skills in data preprocessing, model training, and simple web application development. The solution handles class imbalance and provides a user-friendly interface for model selection and prediction.

### Key Features
- 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, Neural Networks (MLPClassifier)
- ⚖️ **Class Imbalance Handling**: Basic preprocessing with StandardScaler
- 🌐 **Simple Web Interface**: Streamlit application for model selection
- 📊 **Basic Model Evaluation**: Classification reports and accuracy metrics

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
├── dashboard.py                  # Basic dashboard components
├── preprocessing.py              # Data preprocessing utilities
├── logistic_regression.py        # Logistic Regression model
├── random_forest.py              # Random Forest model
├── xgboost_model.py              # XGBoost model
├── neural_network.py             # Neural Network model (MLPClassifier)
├── evaluate_model.py             # Basic model evaluation
├── main.py                       # Main execution script
└── requirrments.txt              # Python dependencies
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
pip install -r requirrments.txt
```

### 3. Download Dataset
```bash
# Download the dataset from Kaggle and place creditcard.csv in the data/ directory
# You can download it from: https://www.kaggle.com/mlg-ulb/creditcardfraud
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
- **Feature Scaling**: StandardScaler for Time and Amount features
- **Data Splitting**: Basic train/test split (80/20)

### Models Implemented

| Model | Implementation |
|-------|----------------|
| Logistic Regression | Scikit-learn LogisticRegression |
| Random Forest | Scikit-learn RandomForestClassifier |
| **XGBoost** | **XGBoost XGBClassifier** |
| Neural Network | Scikit-learn MLPClassifier |

### Evaluation Metrics
- **Classification Report**: Precision, recall, f1-score
- **Basic Accuracy**: Simple accuracy metrics

## 🌐 Web Application

### Streamlit Dashboard
- Model selection interface
- File upload for dataset
- Basic class distribution visualization
- Model training and results display

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms (including MLPClassifier)
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Basic data visualization
- **Imbalanced-learn**: Basic preprocessing

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning Basics**: Training multiple ML models
- **Data Preprocessing**: Feature scaling and data handling
- **Model Selection**: Comparing different algorithms
- **Web Development**: Creating simple interactive applications
- **Version Control**: Git best practices
- **Documentation**: Project presentation

## 🚀 Deployment Options

### Local Deployment
```bash
# Run Streamlit app
streamlit run app.py

# Run main script
python main.py
```

### Cloud Deployment
- **Streamlit Cloud**: Easy deployment of Streamlit apps
- **Heroku**: Containerized application deployment

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

