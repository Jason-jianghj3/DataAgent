"""运行SCADA ETL测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.etl_pipeline import ETLPipeline

p = ETLPipeline()

# 先跑 SCADA hourly agg
r = p.etl_scada_hourly_agg()
status = 'OK' if r['success'] else 'FAIL'
rows = r.get('rows', 0)
elapsed = r.get('elapsed', 0)
error = r.get('error', '')
line = f'  [{status}] dw_scada_hourly_agg: {rows} rows, {elapsed}s'
if error:
    line += f' ({error})'
print(line)

# 如果 hourly agg 成功，再跑 threshold event
if r['success'] and rows > 0:
    r2 = p.etl_scada_threshold_event()
    status2 = 'OK' if r2['success'] else 'FAIL'
    line2 = f'  [{status2}] dw_scada_threshold_event: {r2.get("rows",0)} rows, {r2.get("elapsed",0)}s'
    if r2.get('error'):
        line2 += f' ({r2["error"]})'
    print(line2)
