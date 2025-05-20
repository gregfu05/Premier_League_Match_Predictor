import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "EPL_database.db")
    conn = sqlite3.connect(db_path)
    season_tables = [
        'DataCoUk_Season2017_2018', 'DataCoUk_Season2018_2019', 'DataCoUk_Season2019_2020',
        'DataCoUk_Season2020_2021', 'DataCoUk_Season2021_2022', 'DataCoUk_Season2022_2023',
        'DataCoUk_Season2023_2024', 'DataCoUk_Season2024_2025'
    ]
    db = pd.concat([pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in season_tables], ignore_index=True)
    conn.close()
    return db

def preprocess_data(db):
    db['Referee'] = db['Referee'].str.strip()

    home_win_cols = ['B365H','BWH','IWH','LBH','PSH','WHH','VCH','MaxH','AvgH','PSCH','B365CH','BWCH','IWCH','WHCH','VCCH','MaxCH','AvgCH','BFH','BFCH','1XBH','1XBCH']
    draw_cols     = ['B365D','BWD','IWD','LBD','PSD','WHD','VCD','MaxD','AvgD','PSCD','B365CD','BWCD','IWCD','WHCD','VCCD','MaxCD','AvgCD','BFD','BFCD','1XBD','1XBCD']
    away_win_cols = ['B365A','BWA','IWA','LBA','PSA','WHA','VCA','MaxA','AvgA','PSCA','B365CA','BWCA','IWCA','WHCA','VCCA','MaxCA','AvgCA','BFA','BFCA','1XBA','1XBCA']
    
    db['odds_hw'] = db[home_win_cols].mean(axis=1)
    db['odds_d']  = db[draw_cols].mean(axis=1)
    db['odds_aw'] = db[away_win_cols].mean(axis=1)
    db.drop(columns=home_win_cols + draw_cols + away_win_cols, inplace=True, errors='ignore')

    db.drop(columns=['Div', 'Date', 'FTR'], inplace=True, errors='ignore')

    # Map teams to IDs for convenience but keep original team names for categorical
    teams = pd.unique(db[['HomeTeam', 'AwayTeam']].values.ravel())
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    db['HomeTeam_ID'] = db['HomeTeam'].map(team_to_id)
    db['AwayTeam_ID'] = db['AwayTeam'].map(team_to_id)

    return db

def split_data(db):
    X = db.drop(columns=['FTHG', 'FTAG'])
    y = db[['FTHG', 'FTAG']]

    train_end, val_end = -100, -50
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Include HTR as categorical
    cat_cols = ['HomeTeam', 'AwayTeam', 'Referee', 'HTR']

    # Convert to category dtype
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Drop or preprocess Time (if you want to keep, transform to numeric)
    # Example: extract hour from time string "HH:MM"
    def process_time(df):
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        return df

    X_train = process_time(X_train)
    X_val = process_time(X_val)
    X_test = process_time(X_test)

    # After conversion, you may want to add 'Time' as numeric (if kept)
    # Or drop if you decide it's not helpful
    # For now let's keep it numeric

    return X_train, X_val, X_test, y_train, y_val, y_test, cat_cols


def scale_numeric(X_train, X_val, X_test):
    # Scale only numeric columns, leave categorical untouched
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
    X_val_scaled_num = pd.DataFrame(scaler.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index)
    X_test_scaled_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)

    # Replace numeric columns with scaled, keep categorical as is
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = X_train_scaled_num
    X_val_scaled[numeric_cols] = X_val_scaled_num
    X_test_scaled[numeric_cols] = X_test_scaled_num

    return X_train_scaled, X_val_scaled, X_test_scaled

def get_train_val_test_scaled():
    db = load_data()
    db = preprocess_data(db)
    X_train, X_val, X_test, y_train, y_val, y_test, cat_cols = split_data(db)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_train, X_val, X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, cat_cols
