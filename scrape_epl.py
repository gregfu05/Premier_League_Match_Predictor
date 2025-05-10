#!/usr/bin/env python3
import io
import requests
import pandas as pd
import urllib3
from sqlalchemy import create_engine, text
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# --- CONFIGURATION (EPL only) ---
LEAGUE = 'E0'  # Premier League code
CURRENT_SEASON = '2425'  # 2024/25 season code
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
TABLE_NAME = 'DataCoUk_Season2024_2025'

# Full, ordered list of columns in your DB table
desired_cols = [
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA',
    'BFH', 'BFD', 'BFA', 'PSH', 'PSD', 'PSA',
    'WHH', 'WHD', 'WHA', '1XBH', '1XBD', '1XBA',
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
    'BFEH', 'BFED', 'BFEA', 'B365>2.5', 'B365<2.5',
    'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5',
    'Avg>2.5', 'Avg<2.5', 'BFE>2.5', 'BFE<2.5',
    'AHh', 'B365AHH', 'B365AHA', 'PAHH', 'PAHA',
    'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA',
    'BFEAHH', 'BFEAHA', 'B365CH', 'B365CD', 'B365CA',
    'BWCH', 'BWCD', 'BWCA', 'BFCH', 'BFCD', 'BFCA',
    'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA',
    '1XBCH', '1XBCD', '1XBCA', 'MaxCH', 'MaxCD', 'MaxCA',
    'AvgCH', 'AvgCD', 'AvgCA', 'BFECH', 'BFECD', 'BFECA',
    'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5',
    'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5',
    'BFEC>2.5', 'BFEC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA',
    'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH',
    'AvgCAHA', 'BFECAHH', 'BFECAHA'
]

# Disable warnings if you choose to skip SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def make_retry_session(
    retries=5,
    backoff_factor=1.0,
    status_forcelist=(500, 502, 503, 504),
    session=None
):
    """Create a requests Session that retries on HTTP 5xx errors."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"GET"},
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Set a realistic User-Agent
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    })
    return session

def fetch_epl_df() -> pd.DataFrame:
    """Download and return the EPL CSV for 2024/25 as a DataFrame."""
    session = make_retry_session()
    url = BASE_URL.format(season=CURRENT_SEASON, league=LEAGUE)
    # If SSL is still a problem, you can add verify=False here
    resp = session.get(url, timeout=10, verify=False)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    # Clean column names by removing BOM and any leading/trailing whitespace
    df.columns = df.columns.str.strip().str.replace('ï»¿', '')

    # Parse the Date column (DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    missing = set(desired_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Return only the columns in the correct order
    return df[desired_cols]

def update_database(db_uri: str):
    """Append new EPL rows into your DataCoUk_Season2024_2025 table."""
    engine = create_engine(db_uri, echo=False)
    with engine.begin() as conn:
        # 1. Find the latest stored date (stored as DD/MM/YYYY)
        result = conn.execute(text(f"SELECT MAX(Date) AS max_date FROM {TABLE_NAME}"))
        row = result.fetchone()
        if row and row.max_date:
            last_date = datetime.strptime(row.max_date, "%d/%m/%Y")
        else:
            last_date = datetime(1900, 1, 1)
        print(f"Latest date in {TABLE_NAME}: {last_date.date()}")

        # 2. Fetch new data and filter
        df = fetch_epl_df()
        new_df = df[df['Date'] > last_date].copy()  # Create a copy to avoid SettingWithCopyWarning
        if new_df.empty:
            print("No new EPL rows to append.")
            return

        # 3. Convert Date to string format DD/MM/YYYY
        new_df['Date'] = new_df['Date'].dt.strftime('%d/%m/%Y')

        # 4. Convert numeric columns to appropriate types
        numeric_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                       'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        for col in numeric_cols:
            new_df.loc[:, col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)

        # 5. Convert odds columns to float and handle None values
        odds_cols = [col for col in new_df.columns if any(x in col for x in ['B365', 'BW', 'BF', 'PS', 'WH', '1XB', 'Max', 'Avg', 'BFE', 'P', 'PC'])]
        for col in odds_cols:
            # Convert to numeric, replacing None with NaN
            new_df.loc[:, col] = pd.to_numeric(new_df[col], errors='coerce')
            # Replace NaN with None (which SQLite will convert to NULL)
            new_df.loc[:, col] = new_df[col].where(pd.notnull(new_df[col]), None)

        # 6. Convert DataFrame to list of dictionaries for manual insertion
        records = new_df.to_dict('records')
        
        # 7. Insert records one by one to handle NULL values properly
        for record in records:
            # Create the SQL query with quoted column names and bind parameters
            columns = ', '.join(f'"{k}"' for k in record.keys())
            placeholders = ', '.join([f':{i}' for i in range(len(record))])
            values = {str(i): v for i, v in enumerate(record.values())}
            
            sql = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"
            conn.execute(text(sql), values)
            
        print(f"Appended {len(records)} new rows into {TABLE_NAME}.")

if __name__ == "__main__":
    # Change this URI to point at your database
    DATABASE_URI = "sqlite:///data/EPL_database.db"
    update_database(DATABASE_URI)
