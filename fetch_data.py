#!/usr/bin/env python3
"""Data fetcher: Binance 4h klines (BTC/BNB) + Deribit BTC DVOL

Incrementally updates the CSVs under data/. First run pulls full history.
Deribit DVOL fine-grained data is only retained for ~42 days; older periods
automatically fall back to 12h granularity forward-filled to 4h.

Usage:
    python fetch_data.py                     # incremental update
    python fetch_data.py --start 2021-01-01  # full history from date
"""
import argparse
import time
from pathlib import Path

import requests
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'
BINANCE = 'https://api.binance.com/api/v3/klines'
DERIBIT = 'https://www.deribit.com/api/v2/public/get_volatility_index_data'


def fetch_binance_4h(symbol, fname, start_ts=None):
    """Fetch 4h klines and append to CSV"""
    path = DATA / fname
    if path.exists():
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        last = df['datetime'].iloc[-1]
        start_ms = int((last + pd.Timedelta(hours=4)).timestamp() * 1000)
    else:
        df = pd.DataFrame()
        start_ms = int(pd.Timestamp(start_ts or '2021-01-01').timestamp() * 1000)
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)

    rows = []
    while start_ms <= end_ms:
        r = requests.get(BINANCE, params={'symbol': symbol, 'interval': '4h',
                                          'startTime': start_ms, 'limit': 1000},
                         timeout=30).json()
        if not r:
            break
        rows += r
        start_ms = r[-1][6] + 1
        time.sleep(0.2)
    print(f'[{symbol}] fetched {len(rows)} new bars')

    if rows:
        new = pd.DataFrame(rows)[[0, 1, 2, 3, 4, 5]]
        new.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        new['datetime'] = pd.to_datetime(new['open_time'], unit='ms')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            new[c] = pd.to_numeric(new[c])
        new = new[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df = pd.concat([df, new]).drop_duplicates('datetime').sort_values('datetime')
        df.to_csv(path, index=False)
        print(f'[{symbol}] saved, latest {df["datetime"].iloc[-1]}, {len(df)} rows total')


def fetch_deribit(desc, resolution, start_ts, end_ts):
    cont, total = None, []
    while True:
        params = {'currency': 'BTC', 'start_timestamp': start_ts, 'end_timestamp': end_ts,
                  'resolution': str(resolution)}
        if cont:
            params['continuation'] = cont
        r = requests.get(DERIBIT, params=params, timeout=30).json()
        res = r.get('result', {})
        total += res.get('data', [])
        cont = res.get('continuation')
        if not cont or len(total) > 60000:
            break
        time.sleep(0.15)
    print(f'[DVOL {desc}] {len(total)} rows')
    return total


def fetch_dvol(start_ts=None):
    """DVOL: 12h (older) + hourly (recent) -> resample to 4h, forward-fill"""
    path = DATA / 'deribit_dvol_btc_4h.csv'
    if path.exists():
        old = pd.read_csv(path)
        old['datetime'] = pd.to_datetime(old['datetime'])
        old = old.set_index('datetime')
        anchor = old.index[-1] - pd.Timedelta(hours=12)
    else:
        old = pd.DataFrame()
        anchor = pd.Timestamp(start_ts or '2021-05-01')

    end_ts = int(pd.Timestamp.now().timestamp() * 1000)
    start_ts = int(anchor.timestamp() * 1000)

    coarse = fetch_deribit('12h', 43200, start_ts, end_ts)
    fine = fetch_deribit('fine', 3600, start_ts, end_ts)

    def to_series(rows):
        d = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c'])
        d['datetime'] = pd.to_datetime(d['ts'], unit='ms')
        return d.set_index('datetime')['c'].astype(float)

    s = pd.concat([to_series(coarse), to_series(fine)])
    s = s[~s.index.duplicated(keep='last')].sort_index()
    dvol4 = s.resample('4h').last().ffill()
    if len(old):
        dvol4 = dvol4[dvol4.index > old.index[-1]]

    if len(dvol4) == 0:
        print('[DVOL] no new data')
        return
    new = dvol4.to_frame('dvol').reset_index()
    new.columns = ['datetime', 'dvol']
    if len(old):
        combined = pd.concat([old.reset_index(), new]).sort_values('datetime')
    else:
        combined = new
    combined.to_csv(path, index=False)
    print(f'[DVOL] saved, {len(dvol4)} new rows, latest {dvol4.index[-1]} ({dvol4.iloc[-1]:.2f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (first/full fetch only)')
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    fetch_binance_4h('BTCUSDT', 'btcusdt_4h.csv', args.start)
    fetch_binance_4h('BNBUSDT', 'bnbusdt_4h.csv', args.start)
    fetch_dvol(args.start)
    print('\n=== done ===')
