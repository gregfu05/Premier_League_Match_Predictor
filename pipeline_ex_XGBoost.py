import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.model_selection import train_test_split # Kept for consistency, though not directly used for main split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    """
    Loads data from the EPL_database.db SQLite database.
    Constructs a dynamic path to the database file relative to this script.
    Concatenates data from multiple season tables.
    """
    db_path = os.path.join(os.path.dirname(__file__), "data", "EPL_database.db")
    conn = sqlite3.connect(db_path)

    # Define season tables to load
    season_tables = [
        'DataCoUk_Season2017_2018', 'DataCoUk_Season2018_2019', 'DataCoUk_Season2019_2020',
        'DataCoUk_Season2020_2021', 'DataCoUk_Season2021_2022', 'DataCoUk_Season2022_2023',
        'DataCoUk_Season2023_2024', 'DataCoUk_Season2024_2025' # Future seasons included for schema consistency
    ]
    
    # Load and concatenate data from all specified tables
    db_frames = []
    for table in season_tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            db_frames.append(df)
        except pd.io.sql.DatabaseError:
            print(f"Warning: Table {table} not found in the database. Skipping.")

    if not db_frames:
        raise ValueError("No data loaded. Please check database and table names.")
        
    db = pd.concat(db_frames, ignore_index=True)
    conn.close()
    return db


def add_engineered_features(db: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sophisticated feature engineering including rolling statistics, 
    performance rates, composite indices, and head-to-head statistics.
    
    Features created:
    - Rolling averages for goals, shots, corners, fouls (last 5 games)
    - Win/draw/loss rates (last 5 games)
    - Offensive and defensive composite indices
    - Head-to-head historical performance
    - Team strength differentials
    """
    db = db.reset_index(drop=True)

    # Rolling averages for various match statistics (last 5 games)
    for label, team_col, goal_col in [
        ("avg_goals_L5_home",  "HomeTeam", "FTHG"),
        ("avg_goals_L5_away",  "AwayTeam", "FTAG"),
        ("avg_shots_home_L5",  "HomeTeam", "HS"),
        ("avg_shots_away_L5",  "AwayTeam", "AS"),
        ("avg_shots_ont_home_L5",  "HomeTeam", "HST"),
        ("avg_shots_ont_away_L5",  "AwayTeam", "AST"),
        ("avg_corners_home_L5",     "HomeTeam", "HC"),
        ("avg_corners_away_L5",     "AwayTeam", "AC"),
        ("avg_goals_conceded_home_L5", "HomeTeam", "FTAG"),
        ("avg_goals_conceded_away_L5", "AwayTeam", "FTHG"),
        ("avg_fouls_committed_home_L5", "HomeTeam", "HF"),
        ("avg_fouls_committed_away_L5", "AwayTeam", "AF"),
    ]:
        db[label] = (
            db.groupby(team_col)[goal_col]
              .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

    def last5_rate(frame, team_col, wanted):
        """Calculate win/draw/loss rates over last 5 games"""
        result_series = (frame['FTR'] == wanted).astype(int)
        return (
            result_series.groupby(frame[team_col])
                 .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

    # Performance rates over last 5 games
    db["win_rate_home_L5"]  = last5_rate(db, "HomeTeam", "H")
    db["win_rate_away_L5"]  = last5_rate(db, "AwayTeam", "A")
    db["draw_rate_home_L5"] = last5_rate(db, "HomeTeam", "D")
    db["draw_rate_away_L5"] = last5_rate(db, "AwayTeam", "D")
    db["loss_rate_home_L5"] = last5_rate(db, "HomeTeam", "A")
    db["loss_rate_away_L5"] = last5_rate(db, "AwayTeam", "H")

    # Team strength differential
    db["home_strength_diff"] = (
        (db["avg_goals_L5_home"] - db["avg_goals_conceded_home_L5"])
        - (db["avg_goals_L5_away"] - db["avg_goals_conceded_away_L5"])
    )

    # Composite offensive indices (weighted combination of attacking metrics)
    db["offensive_index_home"] = (
        0.4*db["avg_goals_L5_home"] + 0.3*db["avg_shots_home_L5"] + 0.3*db["avg_shots_ont_home_L5"]
    )
    db["offensive_index_away"] = (
        0.4*db["avg_goals_L5_away"] + 0.3*db["avg_shots_away_L5"] + 0.3*db["avg_shots_ont_away_L5"]
    )
    
    # Composite defensive solidity indices
    db["defensive_solidity_home"] = (
        0.5*db["avg_goals_conceded_home_L5"] +
        0.25*db["avg_fouls_committed_home_L5"] +
        0.25*(1 - db["win_rate_home_L5"])
    )
    db["defensive_solidity_away"] = (
        0.5*db["avg_goals_conceded_away_L5"] +
        0.25*db["avg_fouls_committed_away_L5"] +
        0.25*(1 - db["win_rate_away_L5"])
    )

    # Head-to-head historical performance (last 5 encounters)
    h2h_home, h2h_away = [], []
    for idx, row in db.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        past = db.loc[:idx-1]
        past_h2h = past[((past["HomeTeam"] == home) & (past["AwayTeam"] == away)) |
                        ((past["HomeTeam"] == away) & (past["AwayTeam"] == home))].tail(5)
        if not past_h2h.empty:
            home_goals = past_h2h.apply(
                lambda r: r["FTHG"] if r["HomeTeam"] == home else r["FTAG"], axis=1
            )
            away_goals = past_h2h.apply(
                lambda r: r["FTAG"] if r["HomeTeam"] == home else r["FTHG"], axis=1
            )
            h2h_home.append(home_goals.mean())
            h2h_away.append(away_goals.mean())
        else:
            h2h_home.append(np.nan)
            h2h_away.append(np.nan)

    db["h2h_avg_home_goals_last_5"] = h2h_home
    db["h2h_avg_away_goals_last_5"] = h2h_away

    # Handle Asian odds if present
    if "odds_home_asian" in db.columns:
        db["odds_home_asian"].fillna(db["odds_home_asian"].mean(), inplace=True)
    if "odds_away_asian" in db.columns:
        db["odds_away_asian"].fillna(db["odds_away_asian"].mean(), inplace=True)

    # Fill NaN values in rolling features using team-specific means
    rolling_cols = [c for c in db.columns if "_L5" in c] + [
        "offensive_index_home", "offensive_index_away",
        "defensive_solidity_home", "defensive_solidity_away"
    ]
    for col in rolling_cols:
        if "home" in col:
            db[col] = db.groupby("HomeTeam")[col].transform(lambda s: s.fillna(s.mean()))
        elif "away" in col:
            db[col] = db.groupby("AwayTeam")[col].transform(lambda s: s.fillna(s.mean()))

    # Final NaN fill for head-to-head features
    db["h2h_avg_home_goals_last_5"].fillna(db["h2h_avg_home_goals_last_5"].mean(), inplace=True)
    db["h2h_avg_away_goals_last_5"].fillna(db["h2h_avg_away_goals_last_5"].mean(), inplace=True)

    return db


def preprocess_data(db):
    """
    Preprocesses the raw data.
    - Strips whitespace from 'Referee' names.
    - Averages betting odds and drops original odds columns.
    - Maps 'HomeTeam' and 'AwayTeam' to integer IDs (though OHE is used later).
    - Applies feature engineering to create rolling statistics and composite indices.
    - Drops 'Div' and 'Date' columns.
    """
    # Strip trailing whitespace from referee names for consistency
    if 'Referee' in db.columns:
        db['Referee'] = db['Referee'].str.strip()

    # Define columns for betting odds
    home_win_cols = ['B365H','BWH','IWH','LBH','PSH','WHH','VCH','MaxH','AvgH','PSCH','B365CH','BWCH','IWCH','WHCH','VCCH','MaxCH','AvgCH','BFH','BFCH','1XBH','1XBCH']
    draw_cols     = ['B365D','BWD','IWD','LBD','PSD','WHD','VCD','MaxD','AvgD','PSCD','B365CD','BWCD','IWCD','WHCD','VCCD','MaxCD','AvgCD','BFD','BFCD','1XBD','1XBCD']
    away_win_cols = ['B365A','BWA','IWA','LBA','PSA','WHA','VCA','MaxA','AvgA','PSCA','B365CA','BWCA','IWCA','WHCA','VCCA','MaxCA','AvgCA','BFA','BFCA','1XBA','1XBCA']
    
    # Filter out columns that are not present in the DataFrame to avoid errors
    home_win_cols = [col for col in home_win_cols if col in db.columns]
    draw_cols = [col for col in draw_cols if col in db.columns]
    away_win_cols = [col for col in away_win_cols if col in db.columns]

    # Create betting odds averages if columns exist
    if home_win_cols:
        db['odds_hw'] = db[home_win_cols].mean(axis=1)
    if draw_cols:
        db['odds_d']  = db[draw_cols].mean(axis=1)
    if away_win_cols:
        db['odds_aw'] = db[away_win_cols].mean(axis=1)
    
    # Drop original betting odds columns
    db.drop(columns=home_win_cols + draw_cols + away_win_cols, inplace=True, errors='ignore')

    # Map team names to unique integer IDs.
    # While XGBoost can handle categorical features, OHE is often preferred for clarity and consistency.
    # These IDs are not directly used if OHE is applied to original team name columns.
    if 'HomeTeam' in db.columns and 'AwayTeam' in db.columns:
        teams = pd.unique(db[['HomeTeam', 'AwayTeam']].values.ravel('K'))
        team_to_id = {team: idx for idx, team in enumerate(teams)}
        db['HomeTeam_ID'] = db['HomeTeam'].map(team_to_id)
        db['AwayTeam_ID'] = db['AwayTeam'].map(team_to_id)

    # Apply feature engineering - this is the key addition!
    db = add_engineered_features(db)

    # Drop 'Div' (Division), 'Date', and 'FTR' columns
    # 'Div' might be redundant if data is for a single league.
    # 'Date' is typically handled by chronological splitting or feature engineering (e.g., day of week).
    # 'FTR' is needed for feature engineering but not for final features
    db.drop(columns=['Div', 'Date', 'FTR'], inplace=True, errors='ignore')

    return db


def split_and_encode(db):
    """
    Splits data into training, validation, and test sets (80/10/10).
    One-hot encodes categorical features ('HomeTeam', 'AwayTeam', 'Referee').
    Ensures consistent feature columns across all splits.
    Handles potential NaNs after encoding and selects numeric types.
    """
    db = db.reset_index(drop=True) # Ensure a clean index

    # Define target variables
    target_cols = ['FTHG', 'FTAG']
    if not all(col in db.columns for col in target_cols):
        raise ValueError(f"Target columns {target_cols} not found in DataFrame.")
    
    X = db.drop(columns=target_cols, errors='ignore')
    y = db[target_cols]

    # Percentage-based chronological split (80% train, 10% validation, 10% test)
    n_total = len(X)
    if n_total < 10: # Minimum threshold for a meaningful split
        raise ValueError("Dataset too small for an 80/10/10 split. Need at least 10 samples.")
        
    train_end_idx = int(n_total * 0.8)
    val_end_idx = int(n_total * 0.9)

    X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
    X_val, y_val = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
    X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]

    # Identify categorical columns for one-hot encoding
    cat_cols = ['HomeTeam', 'AwayTeam', 'Referee']
    # Filter cat_cols to only include columns present in X_train
    cat_cols = [col for col in cat_cols if col in X_train.columns]

    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit encoder on training data and transform all sets
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_val = encoder.transform(X_val[cat_cols])
        encoded_test = encoder.transform(X_test[cat_cols])

        encoded_names = encoder.get_feature_names_out(cat_cols)
        
        encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_names, index=X_train.index)
        encoded_val_df = pd.DataFrame(encoded_val, columns=encoded_names, index=X_val.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_names, index=X_test.index)

        # Drop original categorical columns and join encoded ones
        X_train = X_train.drop(columns=cat_cols).join(encoded_train_df)
        # Reindex val/test to match training columns, filling new columns with 0
        # This ensures consistency if some categories are not present in all splits.
        X_val   = X_val.drop(columns=cat_cols).join(encoded_val_df.reindex(columns=encoded_train_df.columns, fill_value=0))
        X_test  = X_test.drop(columns=cat_cols).join(encoded_test_df.reindex(columns=encoded_train_df.columns, fill_value=0))
    else:
        print("Warning: No specified categorical columns found for one-hot encoding.")

    # Ensure only numeric columns are kept (important before scaling)
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Fill any remaining NaNs with 0 (XGBoost can handle NaNs, but explicit filling is often safer for pipelines)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    print("--- NaN Check before scaling ---")
    print(f"NaNs in X_train: {X_train.isna().sum().sum()}")
    print(f"NaNs in X_val: {X_val.isna().sum().sum()}")
    print(f"NaNs in X_test: {X_test.isna().sum().sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Scales numerical features using StandardScaler.
    While XGBoost is not highly sensitive to feature scaling, it can sometimes help
    with convergence or if regularization is used. It also ensures consistency
    if comparing with other models.
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform all sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with original column names and indices
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("--- NaN Check after scaling ---")
    print(f"NaNs in X_train_scaled: {X_train_scaled_df.isna().sum().sum()}")
    print(f"NaNs in X_val_scaled: {X_val_scaled_df.isna().sum().sum()}")
    print(f"NaNs in X_test_scaled: {X_test_scaled_df.isna().sum().sum()}")

    return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df


# --- Add a function to clean column names for XGBoost safety ---
def clean_column_names(df):
    df.columns = (
        df.columns.astype(str)
        .str.replace('[', '_', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '_', regex=False)
        .str.replace('>', '_', regex=False)
        .str.replace(' ', '_')
    )
    return df


def get_train_val_test_data():
    """
    Main function to load, preprocess, split, encode, and scale data.
    Returns scaled training, validation, and test sets (X) and corresponding targets (y).
    """
    db = load_data()
    db = preprocess_data(db)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_encode(db)
    
    # Ensure X_train, X_val, X_test are not empty before scaling
    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("One or more data splits (X_train, X_val, X_test) are empty before scaling. Check data and splitting logic.")

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    # Clean column names for XGBoost compatibility
    X_train_scaled = clean_column_names(X_train_scaled)
    X_val_scaled = clean_column_names(X_val_scaled)
    X_test_scaled = clean_column_names(X_test_scaled)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

if __name__ == '__main__':
    # Example of how to use the pipeline
    print("Running data pipeline for XGBoost...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_data()
        print("--- Data Shapes ---")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("Pipeline executed successfully. Data is ready for XGBoost modeling.")
        
        # Further checks
        print("--- Sample Data (First 5 rows of X_train) ---")
        print(X_train.head())
        print("--- Target Data (First 5 rows of y_train) ---")
        print(y_train.head())

    except ValueError as ve:
        print(f"ValueError during pipeline execution: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 