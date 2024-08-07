import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import datetime as dt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNetCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

# קביעת נתיב הקובץ
file_path = r"C:\Users\lior vauonuo\car_proj\car_data.csv"

# קריאת הקובץ ל-DataFrame
df = pd.read_csv(file_path)

# פונקציה לבחירת פיצ'רים באמצעות forward selection
def forward_selection(X, y, significance_level=0.05):
    initial_features = []
    remaining_features = list(X.columns)
    
    while remaining_features:
        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[initial_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        
        min_p_value = new_pval.min()
        
        if min_p_value < significance_level:
            best_feature = new_pval.idxmin()
            initial_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    
    return initial_features

# התאמת שם עמודת היעד
X = df.drop(columns=['Price'])
y = df['Price']

# בודק אם יש ערכים חסרים ומסיר אותם
X = X.dropna()
y = y.loc[X.index]

# שימוש בפונקציה לבחירת פיצ'רים
selected_features = forward_selection(X, y)

# שימוש בפיצ'רים שנבחרו
X_selected = X[selected_features]

# סטנדרטיזציה של הנתונים
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# בניית מודל Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# חיזוי באמצעות 10-fold cross-validation
y_pred_cv = cross_val_predict(rf_model, X_scaled, y, cv=10)

import pickle
pickle.dump(rf_model, open("trained_model.pkl","wb"))
rf_model = pickle.load(open("trained_model.pkl","rb"))