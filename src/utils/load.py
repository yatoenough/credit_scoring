import pandas as pd
from sqlalchemy import create_engine


def load_sql(conn_str: str)  -> pd.DataFrame:
    engine = create_engine(conn_str)
    df = pd.read_sql("SELECT * FROM scoring;", engine)
    return df
