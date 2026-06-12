"""运行首次ETL"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.etl_pipeline import ETLPipeline

p = ETLPipeline()
results = p.run_all()

print()
for name, r in results.items():
    status = 'OK' if r['success'] else 'FAIL'
    rows = r.get('rows', 0)
    elapsed = r.get('elapsed', 0)
    error = r.get('error', '')
    line = f'  [{status}] {name}: {rows} rows, {elapsed}s'
    if error:
        line += f' ({error})'
    print(line)

ok = sum(1 for r in results.values() if r['success'])
print(f'\n总计: {ok}/{len(results)} 成功')
