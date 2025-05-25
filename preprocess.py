import sqlite3
import pandas as pd

def preprocess_data():
    # Connect to the separate preprocessing database
    conn = sqlite3.connect('preprocess_database.db')

    # Fetch and preprocess data
    df = pd.read_sql_query('SELECT * FROM performance_data', conn)

    # Example preprocessing step
    df['distance'] = df['distance'] * 1.1  # Example: scale distance

    # Save the preprocessed data back to the main database
    with sqlite3.connect('database.db') as conn_main:
        df.to_sql('athletic_data', conn_main, if_exists='append', index=False)

    conn.close()

if __name__ == '__main__':
    preprocess_data()
