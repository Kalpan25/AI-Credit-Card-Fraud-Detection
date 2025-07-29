from preprocessing import load_and_preprocess_data
from random_forest import run_random_forest
from logistic_regression import run_logistic_regression
from neural_network import run_neural_network
from xgboost_model import run_xgboost

# Load and preprocess data once
df = load_and_preprocess_data()

while True:
    print("\nSelect a model to train and evaluate:")
    print("1. Random Forest")
    print("2. Logistic Regression")
    print("3. Neural Network")
    print("4. XGBoost")
    print("5. Exit")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        run_random_forest(df)
    elif choice == '2':
        run_logistic_regression(df)
    elif choice == '3':
        run_neural_network(df)
    elif choice == '4':
        run_xgboost(df)
    elif choice == '5':
        print("Exiting the program Thank you for using our program!")
        break
    else:
        print("Invalid choice. Please try again.")