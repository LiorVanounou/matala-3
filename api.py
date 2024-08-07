from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# טעינת המודל המאומן
model = pickle.load(open("trained_model.pkl", "rb"))

# נתוני הפיצ'רים שנבחרו
selected_features = ['manufactor', 'year', 'model', 'gear', 'engine', 'engine-type', 'prev-ownership', 'curr-ownership', 'city']

# טעינת סקלר (אם השתמשת בסקלר לאימונים)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # קבלת הנתונים מהטופס
    input_data = {
        'manufactor': request.form['manufactor'],
        'year': request.form['year'],
        'model': request.form['model'],
        'gear': request.form['gear'],
        'engine': request.form['engine'],
        'engine-type': request.form['engine-type'],
        'prev-ownership': request.form['prev-ownership'],
        'curr-ownership': request.form['curr-ownership'],
        'city': request.form['city']
    }
    
    # המרת הנתונים ל-DataFrame
    input_df = pd.DataFrame([input_data])
    
    # ביצוע סטנדרטיזציה על הנתונים (בהנחה שהסקלר אומן עם אותם נתונים)
    input_scaled = scaler.transform(input_df[selected_features])
    
    # חיזוי מחיר הרכב
    prediction = model.predict(input_scaled)
    
    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
