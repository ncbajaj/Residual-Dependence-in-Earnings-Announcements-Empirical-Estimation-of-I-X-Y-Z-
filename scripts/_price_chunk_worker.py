"""
Worker process for one chunk of tickers. Run as a fresh subprocess per
chunk (see build_dataset_cache_chunked.py) so each chunk gets a clean
yfinance/requests connection pool -- avoids the progressive slowdown
seen when one long-lived process makes hundreds of sequential
yf.download() calls.

argv: <input_pickle> <output_pickle>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from main import stage_price_features, stage_abnormal_returns


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    df = pd.read_pickle(in_path)
    df = stage_price_features(df)
    df = stage_abnormal_returns(df)
    df.to_pickle(out_path)


if __name__ == "__main__":
    main()
