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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
def prepare_data(df):
    # מחיקת רווחים והסרת ירודים משמות העמודות
    df.columns = df.columns.str.strip().str.replace('\r\n', '').str.replace('\n', '').str.replace('\r', '')
    
    # הסרת עמודות לפי שם
    df.drop(columns=['Area', 'Cre_date', 'Repub_date', 'Supply_score', 'Test'], inplace=True)
    
    # נקיון עמודת MANUFACTURE
    df.manufactor = df.manufactor.str.replace('Lexsus', 'לקסוס')
    
    # נקיון עמודת MODEL
    words_to_remove = [
        "\r\n", "החדש", "הדור", "חדשה", 'אאודי', 'סקודה', 'ניסאן', 'לקסוס', 'סוזוקי', 'מאזדה', 'החדשה', r'\(\d+\)', 
        'רנו', 'אלפא רומיאו', 'יונדאי', 'ניסאן', 'סוזוקי', 'טויוטה', 'קיה', 'אאודי', 'סובארו', 'מיצובישי', 'מרצדס', 
        'ב.מ.וו', 'אופל', 'הונדה', 'פולקסווגן', 'שברולט', 'מאזדה', 'וולוו', 'סקודה', 'פורד', 'קרייזלר', 'סיטרואן', 
        "פיג'ו", 'רנו', 'לקסוס', 'דייהטסו', 'מיני', 'אלפא רומיאו'
    ]
    pattern = '|'.join(words_to_remove)
    df.model = df.model.str.replace(pattern, '', regex=True)
    df.model = df.model.str.strip()
    
    # פונקציה להחלפת ערך חסר בעמודת סוג הגיר
    def replace_gear(row, df):
        if pd.isna(row['Gear']) or row['Gear'] == 'לא מוגדר':
            mask = (df['model'] == row['model']) & (df['Year'] == row['Year']) & (df['manufactor'] == row['manufactor'])
            matching_rows = df[mask]
            if len(matching_rows) > 0:
                new_gear_value = matching_rows.iloc[0]['Gear']
                return new_gear_value
        return row['Gear']

    df['Gear'] = df.apply(replace_gear, axis=1, df=df)
    
    # החלפת ערכים לסוג מנוע
    df['Gear'] = df['Gear'].replace({'אוטומט': 'אוטומטית'})
    
    # החלפת ערכים לעמודת capacity_Engine
    replace_dict1 = {
        '132': '1320',
        '105': '1050',
        '110': '1100',
        '150': '1500',
        '13': '1300',
        '90': '900',
        '125': '1250',
        '80': '800',
        '12000': '1200'
    }
    df['capacity_Engine'] = df['capacity_Engine'].replace(replace_dict1, regex=True)
    
    # פונקציה להחלפת ערכים לא תקינים בנפח מנוע
    def fix_capacity_engine(df):
        for index, row in df.iterrows():
            capacity = row['capacity_Engine']
            if pd.notna(capacity) and isinstance(capacity, (int, float)) and len(str(capacity)) < 4:
                mask = (df['Gear'] == row['Gear']) & (df['Year'] == row['Year']) & (df['model'] == row['model'])
                matching_capacity = df.loc[mask, 'capacity_Engine'].values
                if len(matching_capacity) > 0:
                    df.at[index, 'capacity_Engine'] = matching_capacity[0]
        return df

    df = fix_capacity_engine(df)
    df.model = df.model.str.replace(pattern, '', regex=True)
    # החלפת ערכים לעמודת Engine_type
    df['capacity_Engine'] = df['capacity_Engine'].replace(",", '', regex=True)
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce').fillna(0)

    # החלפת ערכים לעמודות Prev_ownership ו-Curr_ownership
    df['Prev_ownership'] = df['Prev_ownership'].replace('לא מוגדר', 'אחר', regex=True)
    df['Curr_ownership'] = df['Curr_ownership'].replace('לא מוגדר', 'אחר', regex=True)
    
    # החלפת ערכים לעמודת City
    replace_dict2 = {
        'jeruslem': 'ירושלים',
        'Rehovot': 'רחובות',
        'haifa': 'חיפה',
        'Rishon LeTsiyon': 'ראשון לציון',
        'פתח תקווה': 'פ"ת',
        'ashdod': 'אשדוד',
        'Tel aviv': 'תל אביב',
        'ראשון לציון': 'ראשון',
        'Tzur Natan': 'צור נתן',
        "נתניה": "נתנייה"
    }
    df['City'] = df['City'].replace(replace_dict2, regex=True)
    
    # החלפת פסיקים בריקות כדי להסירם
    df['Km'] = df['Km'].str.replace(',', '')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['Km'] = df['Km'].replace(0, np.nan)
    # פונקציה להכפלת ערכים קטנים מ-10000 ב-1000
    def multiply_by_1000(x):
        if pd.notna(x) and x < 10000:
            return x * 1000
        else:
            return x

    df['Km'] = df['Km'].apply(multiply_by_1000)
    
    # יצירת OneHotEncoder והחלתו על העמודות
    def one_hot_encode(df, column):
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=[column], inplace=True)
        return df

    columns_to_encode = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'City', 'Color']
    for column in columns_to_encode:
        df = one_hot_encode(df, column)
    
    # פונקציה לשינוי הערכים בעמודת DESCRIPTION
    def update_description(description):
        has_shamur = 'שמור' in description or 'שמורה' in description
        has_chadash = 'חדש' in description or 'חדשה' in description
        has_matzav_metzuian = 'מצב מצויין' in description or 'מצב מעולה' in description or 'מצב מצוין' in description
        has_matzav_tov = 'מצב טוב' in description
        
        if has_shamur and has_chadash:
            return 'שמור וחדש'
        elif has_shamur:
            return 'שמור'
        elif has_chadash:
            return 'חדש'
        elif has_matzav_metzuian:
            return 'מצב מצויין'
        elif has_matzav_tov:
            return 'מצב טוב'
        else:
            return np.nan

    df['Description'] = df['Description'].apply(update_description)
    df = one_hot_encode(df, 'Description')
    

    
    return df
