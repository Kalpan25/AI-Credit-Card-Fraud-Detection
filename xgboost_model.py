import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_xgboost(df):
    x = df.drop("Class", axis=1)
    y = df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    return report
