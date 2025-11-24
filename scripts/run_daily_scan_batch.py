import subprocess
import sys
from datetime import date, timedelta

ranges = [
    (date(2025, 10, 25), date(2025, 10, 31)),
    (date(2025, 11, 1), date(2025, 11, 20)),
]

for start, end in ranges:
    d = start
    while d <= end:
        dt = d.isoformat()
        print('Scanning {}'.format(dt))
        result = subprocess.run([sys.executable, 'scripts/daily_candidate_scan.py', '--as-of', dt, '--bottom-n', '80'])
        if result.returncode != 0:
            print('Scan failed for {}'.format(dt))
            sys.exit(result.returncode)
        d += timedelta(days=1)
