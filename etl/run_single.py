"""运行单个ETL任务"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.etl_pipeline import ETLPipeline

p = ETLPipeline()
results = p.run_all()
r = results.get('dw_workorder_pending', {})
status = 'OK' if r['success'] else 'FAIL'
rows = r.get('rows', 0)
elapsed = r.get('elapsed', 0)
error = r.get('error', '')
line = f'  [{status}] dw_workorder_pending: {rows} rows, {elapsed}s'
if error:
    line += f' ({error})'
print(line)
