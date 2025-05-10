import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    conn = sqlite3.connect("data/EPL_database.db")
    season_tables = [
        'DataCoUk_Season2017_2018', 'DataCoUk_Season2018_2019', 'DataCoUk_Season2019_2020',
        'DataCoUk_Season2020_2021', 'DataCoUk_Season2021_2022', 'DataCoUk_Season2022_2023',
        'DataCoUk_Season2023_2024', 'DataCoUk_Season2024_2025'
    ]
    db = pd.concat([
        pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in season_tables
    ], ignore_index=True)
    conn.close()
    return db


def preprocess_data(db):
    # Strip trailing whitespace from referee names
    db['Referee'] = db['Referee'].str.strip()

    # Create betting odds averages
    home_win_cols = ['B365H','BWH','IWH','LBH','PSH','WHH','VCH','MaxH','AvgH','PSCH','B365CH','BWCH','IWCH','WHCH','VCCH','MaxCH','AvgCH','BFH','BFCH','1XBH','1XBCH']
    draw_cols     = ['B365D','BWD','IWD','LBD','PSD','WHD','VCD','MaxD','AvgD','PSCD','B365CD','BWCD','IWCD','WHCD','VCCD','MaxCD','AvgCD','BFD','BFCD','1XBD','1XBCD']
    away_win_cols = ['B365A','BWA','IWA','LBA','PSA','WHA','VCA','MaxA','AvgA','PSCA','B365CA','BWCA','IWCA','WHCA','VCCA','MaxCA','AvgCA','BFA','BFCA','1XBA','1XBCA']
    db['odds_hw'] = db[home_win_cols].mean(axis=1)
    db['odds_d']  = db[draw_cols].mean(axis=1)
    db['odds_aw'] = db[away_win_cols].mean(axis=1)
    db.drop(columns=home_win_cols + draw_cols + away_win_cols, inplace=True, errors='ignore')

    # Drop date and division (assume already cleaned)
    db.drop(columns=['Div', 'Date'], inplace=True, errors='ignore')

    # Map teams to IDs
    teams = pd.unique(db[['HomeTeam', 'AwayTeam']].values.ravel())
    team_to_id = {team: idx for idx, team in enumerate(teams)}
    db['HomeTeam_ID'] = db['HomeTeam'].map(team_to_id)
    db['AwayTeam_ID'] = db['AwayTeam'].map(team_to_id)

    return db


def split_and_encode(db):
    # Sort chronologically if needed (assumes it is)
    db = db.reset_index(drop=True)

    # Targets
    X = db.drop(columns=['FTHG', 'FTAG'])
    y = db[['FTHG', 'FTAG']]

    # Train/val/test split: last 4 games split off
    train_end, val_end = -4, -2
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # One-hot encode HomeTeam, AwayTeam, Referee
    cat_cols = ['HomeTeam', 'AwayTeam', 'Referee']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train = encoder.fit_transform(X_train[cat_cols])
    encoded_val = encoder.transform(X_val[cat_cols])
    encoded_test = encoder.transform(X_test[cat_cols])

    encoded_names = encoder.get_feature_names_out(cat_cols)
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_names, index=X_train.index)
    encoded_val_df = pd.DataFrame(encoded_val, columns=encoded_names, index=X_val.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_names, index=X_test.index)

    # Drop original cat cols and join
    X_train = X_train.drop(columns=cat_cols).join(encoded_train_df)
    X_val = X_val.drop(columns=cat_cols).join(encoded_val_df)
    X_test = X_test.drop(columns=cat_cols).join(encoded_test_df)

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


# Final callable function

def get_train_val_test_scaled():
    db = load_data()
    db = preprocess_data(db)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_encode(db)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
