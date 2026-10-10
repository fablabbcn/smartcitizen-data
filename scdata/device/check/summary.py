''' Summary of the health checks of a device, as plain json (for storage, e.g. in flows) '''
from pandas import Series

# Intervals kept per column: the first ones in time
MAX_INTERVALS = 10


def flagged_runs(mask):
    ''' Consecutive flagged rows as [(first, last)] timestamps '''
    mask = Series(mask).fillna(False).astype(bool)
    runs, start, last = [], None, None
    for timestamp, flagged in mask.items():
        if flagged and start is None:
            start = timestamp
        if not flagged and start is not None:
            runs.append((start, last))
            start = None
        last = timestamp
    if start is not None:
        runs.append((start, last))
    return runs


def summarise_check(result, data):
    '''
    {column: summary} of a check result. result.data has a __<column> boolean per column.
    Checks that give intervals (gaps) are measured in time: the share of the period in them.
    The others in rows: the share of the column's readings flagged
    '''
    columns = dict()
    period_minutes = (data.index.max() - data.index.min()).total_seconds() / 60 if len(data) else 0
    for flag in result.data.columns:
        column = flag[2:] if flag.startswith('__') else flag
        mask = result.data[flag].fillna(False).astype(bool)
        summary = {'flagged': int(mask.sum())}
        if result.intervals is not None and flag in result.intervals:
            intervals = result.intervals[flag]
            minutes = sum((end - start).total_seconds() for start, end in intervals) / 60
            summary['checked'] = int(len(data))
            summary['minutes'] = round(minutes, 1)
            summary['ratio'] = round(min(minutes / period_minutes, 1), 4) if period_minutes else float(bool(intervals))
        else:
            intervals = flagged_runs(mask)
            checked = int(data[column].notna().sum()) if column in data.columns else int(len(mask))
            summary['checked'] = checked
            summary['ratio'] = round(summary['flagged'] / checked, 4) if checked else 0.0
        summary['intervals'] = [[start.isoformat(), end.isoformat()] for start, end in intervals[:MAX_INTERVALS]]
        summary['more_intervals'] = max(len(intervals) - MAX_INTERVALS, 0)
        columns[column] = summary
    return columns


def empty_health(data):
    return {
        'start': data.index.min().isoformat() if len(data) else None,
        'end': data.index.max().isoformat() if len(data) else None,
        'rows': int(len(data)),
        'checks': [],
    }
