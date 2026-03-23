"""
scripts/parse_optimizer.py

Parses MT5 optimizer results XML (Excel XML format).
Saved from MT5: right-click Optimization Results → Save.

Output:
    data/exports/optimizer_results.csv  — all parameter sets with their metrics

How it's used for training:
    Each parameter set has an overall Sharpe ratio.
    We combine this with monthly market features to create training samples.
    18 parameter sets × 70 months = 1260 training samples vs original 30.
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_optimizer_xml(xml_path: str) -> pd.DataFrame:
    """
    Parse MT5 optimizer results XML into a DataFrame.
    Handles both namespaced and plain Excel XML formats.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

    def find_rows(node):
        # Try with namespace first
        rows = node.findall('.//ss:Row', ns)
        if rows:
            return rows
        # Fallback: no namespace
        return node.findall('.//Row')

    def get_cell_value(cell):
        for tag in ['ss:Data', 'Data']:
            try:
                data = cell.find(tag, ns) if 'ss:' in tag else cell.find(tag)
                if data is not None and data.text:
                    return data.text.strip()
            except Exception:
                pass
        return None

    rows = find_rows(root)
    if not rows:
        raise ValueError("No rows found in XML")

    # Header row
    headers = []
    for cell in (rows[0].findall('ss:Cell', ns) or rows[0].findall('Cell')):
        val = get_cell_value(cell)
        if val:
            headers.append(val)

    # Data rows
    records = []
    for row in rows[1:]:
        cells  = row.findall('ss:Cell', ns) or row.findall('Cell')
        values = []
        for cell in cells:
            val = get_cell_value(cell)
            if val is not None:
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(val)
            else:
                values.append(None)
        if values:
            padded = values + [None] * max(0, len(headers) - len(values))
            records.append(dict(zip(headers, padded[:len(headers)])))

    df = pd.DataFrame(records)
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]

    if 'Sharpe_Ratio' in df.columns:
        df = df.dropna(subset=['Sharpe_Ratio'])

    return df


def detect_param_columns(df: pd.DataFrame) -> list:
    known_metrics = {
        'Pass', 'Result', 'Profit', 'Expected_Payoff', 'Profit_Factor',
        'Recovery_Factor', 'Sharpe_Ratio', 'Custom', 'Equity_DD_%',
        'Trades', 'OnTester'
    }
    return [c for c in df.columns if c not in known_metrics]


def main():
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        candidates = (list(Path('.').glob('*.xml')) +
                      list(Path('reports').glob('*.xml')) +
                      list(Path('data').glob('*.xml')))
        if not candidates:
            print("Usage: python scripts/parse_optimizer.py <results.xml>")
            print("Or place the XML in project root / reports/ / data/")
            return
        xml_path = str(candidates[0])
        print(f"Auto-detected: {xml_path}")

    print(f"Parsing: {xml_path}")
    df = parse_optimizer_xml(xml_path)
    param_cols = detect_param_columns(df)

    print(f"\nParsed {len(df)} parameter combinations")
    print(f"Parameters: {param_cols}")
    print(f"Sharpe range: {df['Sharpe_Ratio'].min():.2f} — {df['Sharpe_Ratio'].max():.2f}")
    print(f"Profit range: ${df['Profit'].min():,.0f} — ${df['Profit'].max():,.0f}")

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/exports/optimizer_results.csv', index=False)
    print(f"\nSaved: data/exports/optimizer_results.csv")

    print("\nTop 5 by Sharpe:")
    show_cols = ['Pass', 'Sharpe_Ratio', 'Profit', 'Equity_DD_%'] + param_cols[:4]
    print(df.nlargest(5, 'Sharpe_Ratio')[show_cols].to_string(index=False))

    return df


if __name__ == '__main__':
    main()
