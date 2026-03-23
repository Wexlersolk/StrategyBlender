"""
parse_backtest.py — reads all MT5 HTML reports from Reports/ subfolders.

Structure:
    Reports/
        Report1/  *.html
        Report2/  *.html
        ...

Output:
    data/exports/backtest_monthly.csv
    data/exports/backtest_deals.csv
"""

import sys, os, glob
sys.path.append('.')

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def parse_report(html_path):
    report_name = os.path.basename(os.path.dirname(html_path))
    try:
        with open(html_path, encoding='utf-16-le', errors='replace') as f:
            content = f.read()
    except Exception:
        with open(html_path, encoding='utf-8', errors='replace') as f:
            content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 2:
        print(f"  WARNING: {report_name} — cannot find 2 tables, skipping")
        return {}, pd.DataFrame(), report_name

    # Parameters
    params = {}
    for row in tables[0].find_all('tr'):
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cells) >= 2 and '=' in cells[-1]:
            key, _, val = cells[-1].partition('=')
            try:
                params[key.strip()] = float(val.strip())
            except ValueError:
                params[key.strip()] = val.strip()

    # Deals
    rows = tables[1].find_all('tr')
    deals_start = None
    for i, row in enumerate(rows):
        cells = [td.get_text(strip=True) for td in row.find_all(['td','th'])]
        if 'Deals' in cells:
            deals_start = i + 2
            break

    if deals_start is None:
        print(f"  WARNING: {report_name} — no Deals section, skipping")
        return params, pd.DataFrame(), report_name

    deal_rows = []
    for row in rows[deals_start:]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cells) < 11: continue
        if cells[4] != 'out' or not cells[2]: continue
        try:
            t = pd.to_datetime(cells[0], format='%Y.%m.%d %H:%M:%S')
            p = float(cells[10].replace(' ','').replace(',',''))
            deal_rows.append({
                'time': t, 'profit': p, 'report': report_name,
                'mmLots': params.get('mmLots', 0),
                'SL1':    params.get('StopLossCoef1', 0),
                'PT1':    params.get('ProfitTargetCoef1', 0),
                'SL2':    params.get('StopLossCoef2', 0),
                'PT2':    params.get('ProfitTargetCoef2', 0),
                'TSA':    params.get('TrailingActCef1', 0),
            })
        except (ValueError, IndexError):
            continue

    deals = pd.DataFrame(deal_rows)
    if not deals.empty:
        deals = deals.set_index('time').sort_index()
    return params, deals, report_name


def compute_monthly_stats(deals, report_name):
    if deals.empty:
        return pd.DataFrame()
    monthly = []
    for period, group in deals.groupby(pd.Grouper(freq='ME')):
        profits = group['profit'].values
        if len(profits) < 2: continue
        mean_p = np.mean(profits)
        std_p  = np.std(profits) + 1e-8
        r = group.iloc[0]
        monthly.append({
            'year_month':   period.strftime('%Y-%m'),
            'report':       report_name,
            'sharpe':       round(float(mean_p / std_p * np.sqrt(240)), 4),
            'total_profit': round(float(profits.sum()), 2),
            'win_rate':     round(float(np.mean(profits > 0)), 4),
            'num_trades':   len(profits),
            'avg_profit':   round(float(mean_p), 2),
            'mmLots': r.get('mmLots', 0),
            'SL1':    r.get('SL1', 0),
            'PT1':    r.get('PT1', 0),
            'SL2':    r.get('SL2', 0),
            'PT2':    r.get('PT2', 0),
            'TSA':    r.get('TSA', 0),
        })
    return pd.DataFrame(monthly)


def main():
    reports_dir = 'Reports'
    if not os.path.isdir(reports_dir):
        print(f"ERROR: '{reports_dir}/' not found. Create it and place HTML reports inside.")
        return

    html_files = sorted(set(
        glob.glob(os.path.join(reports_dir, '**', '*.html'), recursive=True) +
        glob.glob(os.path.join(reports_dir, '*.html'))
    ))

    if not html_files:
        print(f"No HTML files found in {reports_dir}/")
        return

    print(f"Found {len(html_files)} report(s):\n")

    all_deals, all_monthly, summary = [], [], []

    for html_path in html_files:
        report_name = os.path.basename(os.path.dirname(html_path))
        if report_name == 'Reports':
            report_name = os.path.splitext(os.path.basename(html_path))[0]

        print(f"  Parsing: {html_path}")
        params, deals, name = parse_report(html_path)

        if deals.empty:
            print(f"    -> No deals, skipping\n")
            continue

        monthly = compute_monthly_stats(deals, name)
        print(f"    -> {len(deals)} deals | {len(monthly)} months | "
              f"mmLots={params.get('mmLots','?')} "
              f"SL1={params.get('StopLossCoef1','?')} "
              f"PT1={params.get('ProfitTargetCoef1','?')}")

        all_deals.append(deals)
        all_monthly.append(monthly)
        summary.append({
            'report':        name,
            'deals':         len(deals),
            'months':        len(monthly),
            'sharpe_mean':   round(monthly['sharpe'].mean(), 2),
            'total_profit':  round(deals['profit'].sum(), 2),
            'mmLots':        params.get('mmLots', '?'),
            'SL1':           params.get('StopLossCoef1', '?'),
            'PT1':           params.get('ProfitTargetCoef1', '?'),
        })

    if not all_monthly:
        print("\nNo valid reports parsed.")
        return

    combined_monthly = pd.concat(all_monthly, ignore_index=True)
    combined_deals   = pd.concat(all_deals)

    os.makedirs('data', exist_ok=True)
    combined_monthly.to_csv('data/exports/backtest_monthly.csv', index=False)
    combined_deals.to_csv('data/exports/backtest_deals.csv')

    print(f"\n{'─'*60}")
    print(f"SUMMARY")
    print(f"{'─'*60}")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nTotal months : {len(combined_monthly)}")
    print(f"Total deals  : {len(combined_deals)}")
    print(f"Sharpe range : {combined_monthly['sharpe'].min():.2f} to {combined_monthly['sharpe'].max():.2f}")
    print(f"\nSaved to data/exports/backtest_monthly.csv and data/exports/backtest_deals.csv")

    print(f"\nSample (first 8 rows):")
    cols = ['year_month','report','sharpe','total_profit','num_trades','mmLots','SL1','PT1']
    print(combined_monthly[cols].head(8).to_string(index=False))


if __name__ == '__main__':
    main()
