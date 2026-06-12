"""
Agent查询引擎 - 基于LLM Function Calling的多步推理查询

核心能力:
1. LLM自主决定执行什么SQL，支持多轮Tool调用
2. 前一轮结果作为后一轮输入，实现多步推理
3. SQL安全验证，只允许SELECT/WITH
4. 流式SSE输出，实时展示思考过程和执行结果
"""

import json
import time
import os
from typing import Generator, Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
from datetime import datetime
from utils.logger import logger
from utils.serialization import safe_json_dumps, convert_rows_types
from utils.db_executor import execute_query

from services.llm_service import LLMService

from services.semantic_cache import get_semantic_cache
from solutions.core.safety_validator import SQLSafetyValidator, SecurityLevel


MAX_AGENT_ROUNDS = 5
MAX_SQL_ROWS = 200

# 模板注册表：意图关键词 → (报表名, 数据集名)
# LLM调用query_by_template时，根据template_name匹配
TEMPLATE_REGISTRY = {
    # EAM工单相关
    "各部门工单处理情况": ("工单效能看板", "各部门工单处理情况（区块二）"),
    "部门工单处理耗时": ("工单效能看板", "各部门工单处理情况（区块二）"),
    "关键指标汇总": ("工单效能看板", "关键指标汇总（区块一）"),
    "处理时长分布": ("工单效能看板", "处理时长分布图（区块三）"),
    "工单数量统计": ("工单效能看板", "工单数量统计（区块三）"),
    "部门人均效能": ("工单效能看板", "部门人均效能（区块四）"),
    "工单完成情况": ("工单效能看板", "工单完成情况"),
    "待审批工单": ("工单效能看板", "待审批工单列表"),
    "部门列表": ("工单效能看板", "部门下拉框"),
    "工单大类列表": ("工单效能看板", "工单大类下拉框"),
    "工单子类列表": ("工单效能看板", "工单子类下拉框"),
    # WMS库存相关
    "各库区库存汇总": ("库存校对", "wms库存数据"),
    "库存总数": ("库存校对", "wms库存总数"),
    "erp库存汇总": ("库存校对", "erp库存数据"),
    "erp库存总数": ("库存校对", "erp库存总数"),
    "托盘库存": ("库存校对", "wms托盘库存"),
    "未上架库存": ("库存校对", "wms未上架库存"),
    "过渡库存": ("库存校对", "wms过渡库存"),
    # WMS呆滞/近效期
    "呆滞物料": ("呆滞物料数据", "呆滞物料明细"),
    "近效期物料": ("近效期数据", "近效期数据"),
    # WMS其他
    "产品信息": ("产品信息维护", "ds1"),
    "公告栏": ("公告栏编辑", "BULLETIN"),
    "到岗人数": ("到岗人数", "ds1"),
}


AGENT_SYSTEM_PROMPT = """你是一个严格基于数据查询结果回答问题的数据分析Agent。

## 当前时间
今天是{current_date}（{current_weekday}）。所有日期范围必须基于{current_year}年计算，不要使用2023年或2024年。

⚠️ 时间关键词精确定义（必须严格遵守，禁止用DATEADD推算替代）：
- "本周" = {this_week_start}至{current_date}（本周一到今天）
- "上周" = {last_week_start}至{last_week_end}（上周一到上周日）
- "本月" = {this_month_start}至{current_date}（本月1号到今天）
- "上月" = {last_month_start}至{last_month_end}（上月1号到上月最后一天）
- "最近7天" = {recent_7d_start}至{current_date}
- "今年" = {current_year}-01-01至{current_date}

⚠️ 关键：用户说"本周"时必须用{this_week_start}至{current_date}，绝对不能用DATEADD(WEEK,-1,GETDATE())（那是最近7天，不是本周）！
⚠️ 关键：用户说"本月"时必须用{this_month_start}至{current_date}，绝对不能用{last_month_start}至{last_month_end}（那是上月）！
⚠️ 禁止使用DATEADD(WEEK,-1,...)、DATEADD(MONTH,-1,...)等相对时间函数来表示"本周"/"本月"，必须用上面给出的精确日期！

## 绝对禁止（违反任何一条即为不合格输出）
- ❌ 禁止凭空编造任何数据、数字或统计结果
- ❌ 禁止使用"XX"、"约"、"大概"等模糊表述代替具体数字
- ❌ 禁止在未执行SQL查询的情况下给出任何数据性结论
- ❌ 禁止使用你自身知识库中的数据来回答问题，只能使用SQL查询返回的数据

## 核心规则
1. ⚠️ 优先使用execute_sql工具查询DW数仓库宽表（connection_name="DW"）！DW宽表已预计算复杂逻辑，只需简单SELECT，准确率最高
2. 只有以下场景才使用query_by_template：处理时长分布、待审批工单、WMS库存/呆滞/近效期物料
3. 只有当DW宽表也无法满足时，才使用execute_sql查询原始库（EAM/WMS_PROD/ekp）
4. 对于复杂问题，先分析需要哪些数据，分步执行
5. 第一步的结果可以作为第二步的输入
6. 每次只调用一个工具，等待结果后再决定下一步
7. 回答要简洁、数据驱动，用具体数字说话
8. 如果一次查询无法回答，可以分多次查询
9. 如果第一次查询失败，分析错误原因并修改SQL重试，最多重试3次
10. 如果查询返回0条数据，如实说明，建议调整查询条件
11. ⚠️ 重要：execute_sql/query_by_template返回数据后，系统会自动生成图表和总结分析，你只需要调用execute_sql或query_by_template即可
12. ⚠️ SCADA查询规则：当用户查询涉及温度、湿度、压差、压力、流量等洁净区环境参数时，必须使用query_scada工具，不要用execute_sql。query_scada支持多设备对比、阈值分析、趋势分析。

## query_scada SCADA查询规则（涉及温度/湿度/压差等环境参数时必须使用）

当用户查询涉及以下场景时，必须调用query_scada工具：
- 单设备查询：如"纯化间温度"、"A1S115湿度" → analysis_type="raw"
- 阈值分析：如"温度超过22度持续多久"、"湿度低于50%的时间" → analysis_type="threshold"，必须同时传threshold数值（不带单位）和threshold_operator
- 多设备对比：如"纯化间和培养间温度对比"、"A1S115和A1S116压差对比" → analysis_type="comparison"，devices用逗号分隔
- 趋势分析：如"温度变化趋势"、"湿度走势" → analysis_type="trend"

⚠️ 阈值分析关键规则：
- analysis_type="threshold"时，threshold参数必传，只传数字不带单位（如22，不是22℃）
- threshold_operator: "超过/高于"用">"（默认），"低于/小于"用"<"
- 示例："昨天纯化间超过22摄氏度持续了多长时间" → devices="纯化间温度", time_range="昨天", analysis_type="threshold", threshold=22, threshold_operator=">"

devices参数格式：房间名+测量类型
- 单设备: "纯化间温度" 或 "培养间湿度"
- 多设备: "纯化间温度,培养间温度"

## query_by_template模板匹配规则（必须遵守）

当用户查询匹配以下意图时，必须调用query_by_template工具：

| 用户查询意图 | template_name | 说明 |
|-------------|---------------|------|
| 各部门工单处理情况/耗时/效率 | 各部门工单处理情况 | 含CTE计算ProcessDays |
| 部门工单处理耗时统计 | 部门工单处理耗时 | 同上，别名 |
| 工单关键指标(总耗时/平均耗时/人均工单) | 关键指标汇总 | 含效能区间判断 |
| 处理时长分布(3小时内/1天到5天等) | 处理时长分布 | 含ProcessTimeCategory |
| 工单数量统计(按大类) | 工单数量统计 | 按工单大类分组 |
| 部门人均效能 | 部门人均效能 | 含处理时长分类 |
| 工单完成情况(已完成/未完成) | 工单完成情况 | 按流程类型+完成状态 |
| 待审批工单/审批停留 | 待审批工单 | ATWORKFLOWSTATUS表 |
| 各库区库存汇总 | 各库区库存汇总 | ZXJT_WMSXCL视图 |
| 库存总数 | 库存总数 | WMS库存总数 |
| 呆滞物料 | 呆滞物料 | 180天未出库 |
| 近效期物料 | 近效期物料 | 90天内到期 |

query_by_template参数说明：
- template_name: 必填，从上表选择
- start_time/end_time: 用户提到时间时传入，格式YYYY-MM-DD
- dept: 用户提到部门时传入部门代码(如CI/OM)，不是中文名
- flow_type: 用户提到工单子类时传入
- flow_category: 用户提到工单大类时传入

⚠️ 示例：
- "各部门工单处理情况" → query_by_template(template_name="各部门工单处理情况")
- "自控信息部本月工单处理情况" → query_by_template(template_name="各部门工单处理情况", dept="CI", start_time="{this_month_start}", end_time="{current_date}")
- "呆滞物料有哪些" → query_by_template(template_name="呆滞物料")
- "库存汇总" → query_by_template(template_name="各库区库存汇总")

## SQL编写规则（必须严格遵守）
1. SQL必须是SELECT或WITH开头，禁止DDL/DML
2. 字符串拼接必须用CONCAT(col1, '|', col2)，绝对不能用col1 + '|' + col2
3. 工单去重计数必须用 COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE))
4. 不要猜测字段名，只使用下面DW宽表字段说明中出现过的字段
5. ⚠️ 必须使用DW宽表！禁止查询原始表（ATWORKFLOWRECORDS/ATUSERS/ATWORKFLOW等）！DW宽表已预计算所有复杂逻辑，只需简单SELECT
6. ⚠️ DW是MySQL数据库！日期格式化用DATE_FORMAT(字段,'%Y-%m-%d')，取前N条用LIMIT N，不支持TOP和CONVERT(varchar,...)
7. ⚠️ 时间筛选规则：只有用户明确提到时间（如"4月"、"最近7天"、"今年"）才加WHERE时间条件！用户没提时间就绝对不能加时间筛选条件！
8. ⚠️ 时间字段规则：工单明细用ApprovalTime（审批时间），待审批工单用LASTSAVED。时间范围用BETWEEN语法。禁止用DATEADD/WEEK/MONTH等SQL Server函数！
9. 必须调用execute_sql工具执行查询获取数据，不能凭空回答
10. GROUP BY必须包含SELECT中所有非聚合字段
11. 部门筛选用 DEPT_CODE IN ('CI','OM')，字符串值用单引号
12. ⚠️ 查询"最多/最少/哪一天"等极值问题：先按维度GROUP BY统计，再ORDER BY + LIMIT 1取极值行

## 可用数据库连接
- DW: 数仓库（优先使用！预计算的宽表，查询简单准确）
- WMS_PROD: 仓储管理数据库（原始表，仅当DW无对应宽表时使用）
- EAM: 设备管理/工单数据库（原始表，仅当DW无对应宽表时使用）
- ekp: OA系统数据库（原始表，仅当DW无对应宽表时使用）

⚠️ 数据库优先级：DW > 原始库。DW中的宽表已经预计算了复杂逻辑（多表JOIN、CTE、ProcessDays等），查询DW宽表只需要简单SELECT，准确率远高于查原始表！

## 数仓库宽表说明（连接: DW，优先使用！）

### 宽表1: dw_workorder_detail（工单明细宽表 - 最常用！）
已预计算ProcessDays/ProcessHours/ProcessTimeCategory，无需CTE！
- RECCODE: 流程编码
- RECFLOCODE: 流程类型代码
- FLODESC: 流程类型名称（工单子类）
- FLOENTITYDESC: 工单大类
- CREATEDBY: 操作人代码
- EMPLOYEE_NAME: 操作人姓名
- DEPT_CODE: 部门代码（OM/FM/QA/DS/CI/QC/VM/FF/EHS/LG/PD/PM）
- DEPT_NAME: 部门名称
- CreateTime: 节点开始时间
- ApprovalTime: 节点结束时间
- ProcessHours: 处理时长(小时)
- ProcessDays: 处理时长(天)
- ProcessTimeCategory: 处理时长分类（3小时内/3小时到1天/1天到5天/5天到10天/10天到30天/30天到90天/大于90天）
- RECFROMSTATUS: 起始状态

### 宽表2: dw_workorder_completion（工单完成情况宽表）
- wfscode: 工单唯一标识
- WFSFLOCODE: 流程代码
- flow_type: 流程类型（截取前7字符）
- completion_status: 完成状态（已完成/未完成）
- LASTSAVED: 最后更新时间

### 宽表3: dw_workorder_pending（待审批工单宽表）
- CREATEDBY: 创建人代码
- DEPT_CODE: 部门代码
- WFSSTATUS: 当前状态
- noddesc: 审批节点描述
- WFSENTITY: 工单实体
- wfscode: 工单编码
- WFSFLOCODE: 流程类型
- LASTSAVED: 最后更新时间
- CurrentApproverID: 当前审批人ID
- CurrentApprover: 当前审批人
- CurrentDepartment: 当前审批部门
- DaysSinceLastSave: 距上次保存天数

### 宽表4: dw_department（部门维度表）
- dept_code: 部门代码
- dept_name: 部门名称

### 宽表5: dw_flow_category（工单大类维度表）
- flow_category: 工单大类
- dept_code: 部门代码

### 宽表6: dw_flow_type（工单子类维度表）
- flow_type: 工单子类
- FLOENTITYDESC: 工单大类
- dept_code: 部门代码

### 宽表7: dw_inventory_by_zone（各库区库存汇总宽表）
- source: 数据来源（ERP/WMS）
- zone_id: 库区代码
- zone_name: 库区名称
- qty: 库存数量

### 宽表8: dw_material_stagnant（呆滞物料宽表）
- material_type: 类型（物料/物资/B、C类物资）
- sku_descr: 物料描述
- sum_qty: 数量
- lot_att06: 自编批号
- alternate_sku3: 物料代码
- uom_descr: 单位
- warehouse_id: 仓库代码
- warehouse_name: 仓库名称
- days_no_out: 未出库天数

### 宽表9: dw_material_expiry（近效期物料宽表）
- material_type: 类型
- expiry_date: 近效期日期
- lot_att06: 自编批号
- alternate_sku3: 物料代码
- sku_descr: 物料描述
- uom_descr: 单位
- sum_qty: 数量
- days_to_expiry: 距到期天数

### 宽表10: dw_inspection_record（验收记录宽表）
- order_date: 订单时间
- asn_no: ASN单号
- doc_type: 单据类型
- zone_name: 库区
- material_name: 物料名称
- material_code: 物料代码
- production_batch: 来厂批号/生产批号
- custom_batch: 自编批号
- quality_status: 质量状态
- quantity: 数量
- warehouse_keeper: 库管员
- receiver: 接收人
- reviewer: 复核人
- remark: 备注
- is_anomaly: 是否异常(1=异常,0=正常)

### 宽表11: dw_scada_hourly_agg（SCADA小时聚合宽表）
- tag_name: 测点标签
- room_name: 房间名
- measure_type: 测量类型（温度/湿度/压差）
- stat_hour: 统计小时
- avg_value: 小时均值
- max_value: 小时最大值
- min_value: 小时最小值
- sample_count: 采样点数

### 宽表12: dw_scada_threshold_event（SCADA阈值超标事件宽表）
- tag_name: 测点标签
- room_name: 房间名
- measure_type: 测量类型
- threshold_value: 阈值
- threshold_operator: 比较运算符(>/</>=/<=)
- start_time: 超标开始时间
- end_time: 超标结束时间
- duration_minutes: 持续分钟数
- max_value: 超标期间最大值
- avg_value: 超标期间均值
- stat_date: 统计日期

## DW宽表查询示例（极简SQL，无需CTE/JOIN！）

⚠️ 以下查询全部使用 connection_name="DW"！DW是MySQL数据库，用LIMIT不用TOP，用DATE_FORMAT不用CONVERT！

CI部门完成工单数量最多的员工：
```sql
SELECT EMPLOYEE_NAME, COUNT(*) AS 工单数量
FROM dw_workorder_detail
WHERE DEPT_CODE = 'CI'
GROUP BY EMPLOYEE_NAME
ORDER BY 工单数量 DESC
LIMIT 1
```

四月中处理工单数量最多的一天：
```sql
SELECT DATE_FORMAT(ApprovalTime, '%Y-%m-%d') AS 日期,
       COUNT(*) AS 工单数量
FROM dw_workorder_detail
WHERE ApprovalTime BETWEEN '2026-04-01' AND '2026-04-30 23:59:59'
GROUP BY DATE_FORMAT(ApprovalTime, '%Y-%m-%d')
ORDER BY 工单数量 DESC
LIMIT 1
```

各部门工单处理情况：
```sql
SELECT DEPT_CODE AS 部门代码, DEPT_NAME AS 部门名称,
       COUNT(*) AS 工单数量, ROUND(AVG(ProcessDays), 3) AS 平均处理天数
FROM dw_workorder_detail
GROUP BY DEPT_CODE, DEPT_NAME
ORDER BY 平均处理天数 ASC
```

处理时长分布：
```sql
SELECT ProcessTimeCategory AS 处理时长分类, COUNT(*) AS 工单数量
FROM dw_workorder_detail
GROUP BY ProcessTimeCategory
ORDER BY 工单数量 DESC
```

待审批工单列表：
```sql
SELECT CurrentApprover AS 审批人, WFSFLOCODE AS 工单类型,
       DaysSinceLastSave AS 停留天数
FROM dw_workorder_pending
ORDER BY 停留天数 DESC
```

工单完成情况：
```sql
SELECT flow_type AS 工单类型, completion_status AS 完成状态, COUNT(*) AS 工单数量
FROM dw_workorder_completion
GROUP BY flow_type, completion_status
ORDER BY 工单类型, 完成状态
```

按日期统计工单数量（用户提到时间时才加时间条件）：
```sql
SELECT DATE_FORMAT(ApprovalTime, '%Y-%m-%d') AS 日期,
       COUNT(*) AS 工单数量
FROM dw_workorder_detail
WHERE ApprovalTime BETWEEN '2026-04-01' AND '2026-04-30 23:59:59'
GROUP BY DATE_FORMAT(ApprovalTime, '%Y-%m-%d')
ORDER BY 工单数量 DESC
```

按员工统计工单数量：
```sql
SELECT EMPLOYEE_NAME AS 员工姓名, DEPT_CODE AS 部门代码,
       COUNT(*) AS 工单数量
FROM dw_workorder_detail
WHERE DEPT_CODE = 'CI'
GROUP BY EMPLOYEE_NAME, DEPT_CODE
ORDER BY 工单数量 DESC
```

按工单大类统计：
```sql
SELECT FLOENTITYDESC AS 工单大类,
       COUNT(*) AS 工单数量
FROM dw_workorder_detail
GROUP BY FLOENTITYDESC
ORDER BY 工单数量 DESC
```

按部门筛选工单：
```sql
SELECT DEPT_CODE AS 部门代码, DEPT_NAME AS 部门名称,
       FLOENTITYDESC AS 工单大类,
       COUNT(*) AS 工单数量
FROM dw_workorder_detail
WHERE DEPT_CODE = 'CI'
GROUP BY DEPT_CODE, DEPT_NAME, FLOENTITYDESC
ORDER BY 工单数量 DESC
```

待审批工单按部门统计：
```sql
SELECT DEPT_CODE AS 部门代码, COUNT(*) AS 待审批数量,
       ROUND(AVG(DaysSinceLastSave), 2) AS 平均停留天数
FROM dw_workorder_pending
WHERE CurrentApproverID IS NOT NULL AND CurrentApproverID <> ''
GROUP BY DEPT_CODE
ORDER BY 待审批数量 DESC
```

待审批工单按审批人统计：
```sql
SELECT CurrentApprover AS 审批人, CurrentDepartment AS 部门, COUNT(*) AS 待审批数量,
       ROUND(AVG(DaysSinceLastSave), 2) AS 平均停留天数
FROM dw_workorder_pending
WHERE CurrentApproverID IS NOT NULL AND CurrentApproverID <> ''
GROUP BY CurrentApprover, CurrentDepartment
ORDER BY 待审批数量 DESC
```

各库区库存汇总（ERP vs WMS对比）：
```sql
SELECT source AS 来源, zone_name AS 库区, qty AS 数量
FROM dw_inventory_by_zone
ORDER BY zone_id, source
```

呆滞物料查询：
```sql
SELECT material_type AS 类型, sku_descr AS 物料名称, sum_qty AS 数量
FROM dw_material_stagnant
ORDER BY sum_qty DESC
```

近效期物料查询：
```sql
SELECT material_type AS 类型, sku_descr AS 物料名称, expiry_date AS 近效期, sum_qty AS 数量
FROM dw_material_expiry
ORDER BY days_to_expiry ASC
```

验收异常记录：
```sql
SELECT order_date, material_name, quality_status, quantity, zone_name
FROM dw_inspection_record
WHERE is_anomaly = 1
ORDER BY order_date DESC
```

SCADA温度超标事件：
```sql
SELECT room_name, start_time, end_time, duration_minutes, max_value
FROM dw_scada_threshold_event
WHERE measure_type = '温度' AND threshold_operator = '>' AND threshold_value = 22
ORDER BY stat_date DESC
```

SCADA小时聚合查询：
```sql
SELECT room_name, stat_hour, avg_value, max_value, min_value
FROM dw_scada_hourly_agg
WHERE room_name = '纯化间' AND measure_type = '温度'
  AND stat_hour BETWEEN '2026-06-08 00:00' AND '2026-06-08 23:59'
ORDER BY stat_hour
```

查询部门列表：
```sql
SELECT dept_code AS 部门代码, dept_name AS 部门名称
FROM dw_department
ORDER BY dept_code
```

部门代码与名称对照表（筛选部门时必须用DEPT_CODE，不能用中文名）：
- OM = 运行保障部
- FM = 设备管理部
- QA = 质量保证部
- DS = 原液生产部
- CI = 自控信息部
- QC = 质量控制部
- VM = 验证管理部
- FF = 制剂生产部
- EHS = 安全环保部
- LG = 物控部
- PD = 采购部
- PM = 生产管理办公室

当用户说"运行保障部"时，必须用 DEPT_CODE = 'OM'

## 查询意图→DW宽表映射规则（必须遵守）

⚠️ 所有工单相关查询必须使用DW宽表，禁止查询原始表！

| 用户意图 | 必须使用的宽表 | 连接 |
|----------|---------------|------|
| 工单数量/排名/耗时/员工排名 | dw_workorder_detail | DW |
| 待审批/待处理/审批停留 | dw_workorder_pending | DW |
| 工单完成情况 | dw_workorder_completion | DW |
| 处理时长分布 | dw_workorder_detail(ProcessTimeCategory) | DW |
| 部门列表 | dw_department | DW |
| 工单大类 | dw_flow_category | DW |
| 工单子类 | dw_flow_type | DW |
| 库存/库区 | dw_inventory_by_zone | DW |
| 呆滞物料 | dw_material_stagnant | DW |
| 近效期物料 | dw_material_expiry | DW |
| 验收记录 | dw_inspection_record | DW |
| SCADA温度/湿度/压差 | dw_scada_hourly_agg | DW |
| SCADA超标事件 | dw_scada_threshold_event | DW |

⚠️ 关键：工单查询必须使用DW宽表（connection_name="DW"），宽表已预计算ProcessDays/ProcessHours/ProcessTimeCategory，禁止使用CTE或JOIN原始表！

DW宽表查询示例（直接简单SELECT，无需CTE/JOIN）：

各部门工单处理情况：
```sql
SELECT DEPT_CODE AS 部门代码, DEPT_NAME AS 部门名称,
       COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS 工单数量,
       ROUND(SUM(ProcessDays), 3) AS 总处理天数,
       ROUND(AVG(ProcessDays), 3) AS 平均处理天数,
       COUNT(DISTINCT CREATEDBY) AS 处理人数
FROM dw_workorder_detail
WHERE DEPT_CODE IN ('CI','OM')
  AND ApprovalTime BETWEEN '2025-05-01' AND '2025-05-31 23:59:59'
GROUP BY DEPT_CODE, DEPT_NAME
ORDER BY 平均处理天数 ASC
```

单个部门员工工单处理明细：
```sql
SELECT EMPLOYEE_NAME AS 员工姓名,
       COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS 工单数量,
       ROUND(AVG(ProcessDays), 3) AS 平均处理天数
FROM dw_workorder_detail
WHERE DEPT_CODE = 'CI'
  AND ApprovalTime BETWEEN '2025-05-01' AND '2025-05-31 23:59:59'
GROUP BY EMPLOYEE_NAME
ORDER BY 平均处理天数 ASC
```

⚠️ 重要规则：
- 工单去重计数必须用 COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE))
- 时间字段用 ApprovalTime（节点结束时间），不用 LASTSAVED
- 员工字段用 EMPLOYEE_NAME，不用 CREATEDBY 或 USRDESC
- 部门筛选用 DEPT_CODE IN ('CI','OM')
- 禁止查询原始表（ATWORKFLOWRECORDS/ATUSERS/ATWORKFLOW），必须查DW宽表
- 禁止使用CTE/WITH子句，DW宽表已预计算所有字段

{sql_examples}
"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_by_template",
            "description": "使用帆软报表预定义SQL模板查询数据。当用户查询匹配已知模板时优先使用此工具，可保证查询结果与帆软报表完全一致。支持参数：start_time/end_time(时间范围)、dept(部门代码如CI/OM)、flow_type(工单子类)、flow_category(工单大类)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "description": "模板名称，必须是以下之一：各部门工单处理情况、部门工单处理耗时、关键指标汇总、处理时长分布、工单数量统计、部门人均效能、工单完成情况、待审批工单、部门列表、工单大类列表、工单子类列表、各库区库存汇总、库存总数、erp库存汇总、erp库存总数、托盘库存、未上架库存、过渡库存、呆滞物料、近效期物料、产品信息、公告栏、到岗人数",
                        "enum": [
                            "各部门工单处理情况", "部门工单处理耗时", "关键指标汇总",
                            "处理时长分布", "工单数量统计", "部门人均效能",
                            "工单完成情况", "待审批工单", "部门列表",
                            "工单大类列表", "工单子类列表",
                            "各库区库存汇总", "库存总数", "erp库存汇总", "erp库存总数",
                            "托盘库存", "未上架库存", "过渡库存",
                            "呆滞物料", "近效期物料",
                            "产品信息", "公告栏", "到岗人数"
                        ]
                    },
                    "start_time": {
                        "type": "string",
                        "description": "开始时间，格式YYYY-MM-DD，用户未提时间则不传"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "结束时间，格式YYYY-MM-DD，用户未提时间则不传"
                    },
                    "dept": {
                        "type": "string",
                        "description": "部门代码，多个用逗号分隔，如CI,OM。用户未提部门则不传"
                    },
                    "flow_type": {
                        "type": "string",
                        "description": "工单子类，多个用逗号分隔。用户未提则不传"
                    },
                    "flow_category": {
                        "type": "string",
                        "description": "工单大类，多个用逗号分隔。用户未提则不传"
                    }
                },
                "required": ["template_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "执行SQL查询并返回结果。只能执行SELECT语句，禁止DDL/DML操作。connection_name指定数据库连接：DW(数仓库宽表，优先使用！)、WMS_PROD(仓储)、EAM(设备管理)、ekp(OA)。⚠️ 优先使用DW连接查询宽表，准确率远高于原始表！",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的SQL查询语句，必须是SELECT或WITH开头"
                    },
                    "connection_name": {
                        "type": "string",
                        "description": "数据库连接名：DW(数仓库，优先)、WMS_PROD、EAM、ekp",
                        "enum": ["DW", "WMS_PROD", "EAM", "ekp"]
                    }
                },
                "required": ["sql", "connection_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_scada",
            "description": "查询SCADA/Historian实时监控数据（温度、湿度、压差、压力、流量等）。当用户查询涉及洁净区环境参数、设备实时数据、温度/湿度/压差对比、超标持续时长时必须使用此工具，不要用execute_sql。",
            "parameters": {
                "type": "object",
                "properties": {
                    "devices": {
                        "type": "string",
                        "description": "要查询的设备描述，格式为'房间+测量类型'，多个用逗号分隔。如'纯化间温度'、'培养间湿度'、'纯化间温度,培养间温度'"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "时间范围描述，如'昨天'、'本周'、'最近3天'、'2026-06-01到2026-06-03'。默认最近1天"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "分析类型：raw(原始数据统计)、threshold(阈值超标分析，必须同时传threshold)、comparison(多设备对比)、trend(趋势变化分析)",
                        "enum": ["raw", "threshold", "comparison", "trend"]
                    },
                    "threshold": {
                        "type": "number",
                        "description": "阈值数值（不带单位），当analysis_type=threshold时必传。如用户说'超过22度'则传22，'低于40%湿度'则传40"
                    },
                    "threshold_operator": {
                        "type": "string",
                        "description": "阈值比较方向：'>'表示超过/高于（默认），'<'表示低于/小于",
                        "enum": [">", "<"]
                    }
                },
                "required": ["devices", "analysis_type"]
            }
        }
    }
]


@dataclass
class AgentStep:
    step: int
    tool_name: str
    tool_input: Dict
    result: Any = None
    error: str = ""


class AgentQueryEngine:
    def __init__(self):
        self.llm_service: Optional[LLMService] = None
        self.safety_validator = SQLSafetyValidator(strict_mode=True)
        self._sql_examples: str = ""
        self._template_store: Dict[str, Dict] = {}  # template_name → {sql, connection, params}
        self._init_components()

    def _init_components(self):
        try:
            self.llm_service = LLMService()
            logger.info("[AgentQueryEngine] LLM服务初始化成功")
        except Exception as e:
            logger.error(f"[AgentQueryEngine] LLM服务初始化失败: {e}")

        self._load_sql_examples()
        self._load_template_store()

    def _load_sql_examples(self):
        try:
            config_path = Path(__file__).parent.parent.parent / "report_config.json"
            if not config_path.exists():
                self._sql_examples = ""
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            examples = []
            reports = config_data.get('reports', {})
            for rpt_name, rpt_info in reports.items():
                datasets = rpt_info.get('datasets', [])
                for ds in datasets:
                    sql = ds.get('sql_template', '')
                    conn = ds.get('connection', '')
                    ds_name = ds.get('name', '')
                    if sql and not sql.startswith('dbo.'):
                        # 清理帆软${if(...)}语法，转为可读的SQL注释
                        clean_sql = self._clean_fanruan_sql(sql)
                        examples.append(
                            f"-- 报表: {rpt_name} | 数据集: {ds_name} | 连接: {conn}\n{clean_sql}"
                        )

            if examples:
                self._sql_examples = "\n\n参考SQL示例（来自report_config.json，编写SQL时必须使用相同的表名和字段名）：\n\n" + "\n\n".join(
                    f"```sql\n{ex}\n```" for ex in examples[:20]
                )
            else:
                self._sql_examples = ""
        except Exception as e:
            logger.warning(f"[AgentQueryEngine] 加载SQL示例失败: {e}")
            self._sql_examples = ""

    @staticmethod
    def _clean_fanruan_sql(sql: str) -> str:
        """
        清理帆软${if(...)}语法，转为可读的SQL条件注释
        让LLM理解参数化逻辑，而不是看到一堆${if()}噪音
        """
        import re

        # 通用处理：替换所有${if(...)}块为可读注释
        # 先处理包含特定关键词的${if()}块
        result_lines = []
        for line in sql.split('\n'):
            if '${if(' in line:
                # 判断是哪种参数
                if 'start_time' in line or 'end_time' in line:
                    result_lines.append('  -- 时间条件: AND ApprovalTime BETWEEN @start_time AND @end_time (未指定时间则不加)')
                elif 'dept' in line and 'USRMRC' in line:
                    result_lines.append('  -- 部门条件: AND AU.USRMRC IN (@dept) (未指定部门则不加)')
                elif 'flow_type' in line and 'FLODESC' in line:
                    result_lines.append('  -- 工单子类条件: AND AF.FLODESC IN (@flow_type) (未指定则不加)')
                elif 'flow_category' in line:
                    result_lines.append('  -- 工单大类条件: AND AF.FLOENTITYDESC IN (@flow_category) (未指定则不加)')
                elif 'dept' in line:
                    result_lines.append('  -- 部门条件: AND AU.USRMRC IN (@dept) (未指定部门则不加)')
                elif 'flow_type' in line:
                    result_lines.append('  -- 工单子类条件: AND AF.FLODESC IN (@flow_type) (未指定则不加)')
                else:
                    # 其他${if()}块直接移除
                    pass
            else:
                result_lines.append(line)

        sql = '\n'.join(result_lines)

        # 清理空行
        sql = re.sub(r'\n\s*\n\s*\n', '\n\n', sql)

        return sql.strip()

    def _load_template_store(self):
        """从report_config.json加载原始SQL模板到_template_store，供query_by_template使用"""
        try:
            config_path = Path(__file__).parent.parent.parent / "report_config.json"
            if not config_path.exists():
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            reports = config_data.get('reports', {})
            for template_name, (rpt_name, ds_name) in TEMPLATE_REGISTRY.items():
                rpt_info = reports.get(rpt_name, {})
                datasets = rpt_info.get('datasets', [])
                for ds in datasets:
                    if ds.get('name') == ds_name:
                        sql = ds.get('sql_template', '')
                        conn = ds.get('connection', '')
                        params = ds.get('params', [])
                        if sql and not sql.startswith('dbo.'):
                            self._template_store[template_name] = {
                                'sql': sql,
                                'connection': conn,
                                'params': params,
                                'report': rpt_name,
                                'dataset': ds_name,
                            }
                        break

            # ===== 用DW宽表简单SQL覆盖工单相关模板 =====
            self._override_with_dw_templates()

            logger.info(
                f"[AgentQueryEngine] 模板仓库加载完成: {len(self._template_store)}/{len(TEMPLATE_REGISTRY)} 个模板"
            )
        except Exception as e:
            logger.warning(f"[AgentQueryEngine] 加载模板仓库失败: {e}")

    def _override_with_dw_templates(self):
        """用DW宽表的简单SQL覆盖工单相关模板，连接改为DW"""
        dw_templates = {
            "各部门工单处理情况": {
                "sql": "SELECT DEPT_CODE, DEPT_NAME, COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS order_count, ROUND(SUM(ProcessDays), 3) AS total_process_days, ROUND(AVG(ProcessDays), 3) AS avg_process_days, COUNT(DISTINCT CREATEDBY) AS user_count FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} GROUP BY DEPT_CODE, DEPT_NAME ORDER BY order_count DESC",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "部门工单处理耗时": {
                "sql": "SELECT DEPT_CODE, DEPT_NAME, COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS order_count, ROUND(SUM(ProcessDays), 3) AS total_process_days, ROUND(AVG(ProcessDays), 3) AS avg_process_days, COUNT(DISTINCT CREATEDBY) AS user_count FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} GROUP BY DEPT_CODE, DEPT_NAME ORDER BY avg_process_days DESC",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "关键指标汇总": {
                "sql": "SELECT COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS 工单数量, ROUND(SUM(ProcessDays), 3) AS 总耗时, ROUND(AVG(ProcessDays), 3) AS 平均耗时, COUNT(DISTINCT CREATEDBY) AS 处理人数, ROUND(COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) * 1.0 / NULLIF(COUNT(DISTINCT CREATEDBY), 0), 3) AS 人均工单 FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} ${flow_type_filter}",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "处理时长分布": {
                "sql": "SELECT ProcessTimeCategory, COUNT(*) AS count, ROUND(COUNT(*) * 100.0 / NULLIF(COUNT(*) OVER(), 0), 1) AS percentage FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} GROUP BY ProcessTimeCategory ORDER BY FIELD(ProcessTimeCategory, '3小时内', '3小时到1天', '1天到5天', '5天到10天', '10天到30天', '30天到90天', '大于90天', '未分类')",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "工单数量统计": {
                "sql": "SELECT FLOENTITYDESC AS 工单大类, COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS 工单数量 FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} GROUP BY FLOENTITYDESC ORDER BY 工单数量 DESC",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "部门人均效能": {
                "sql": "SELECT DEPT_CODE, DEPT_NAME, EMPLOYEE_NAME, COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE)) AS order_count, ROUND(AVG(ProcessDays), 3) AS avg_process_days FROM dw_workorder_detail WHERE 1=1 ${time_filter} ${dept_filter} GROUP BY DEPT_CODE, DEPT_NAME, EMPLOYEE_NAME ORDER BY DEPT_CODE, order_count DESC",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "工单完成情况": {
                "sql": "SELECT flow_type, completion_status, COUNT(DISTINCT CONCAT(wfscode, '|', WFSFLOCODE)) AS 工单数量 FROM dw_workorder_completion WHERE 1=1 ${time_filter} ${flow_type_filter} GROUP BY flow_type, completion_status ORDER BY flow_type, completion_status",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type", "flow_category"],
            },
            "待审批工单": {
                "sql": "SELECT CREATEDBY, DEPT_CODE, WFSSTATUS, noddesc, WFSENTITY, wfscode, WFSFLOCODE, LASTSAVED, CurrentApproverID, CurrentApprover, CurrentDepartment, DaysSinceLastSave FROM dw_workorder_pending WHERE CurrentApproverID IS NOT NULL AND CurrentApproverID <> '' ${dept_filter} ORDER BY DaysSinceLastSave DESC",
                "connection": "DW",
                "params": ["start_time", "end_time", "dept", "flow_type"],
            },
            "部门列表": {
                "sql": "SELECT dept_code, dept_name FROM dw_department ORDER BY dept_code",
                "connection": "DW",
                "params": [],
            },
            "工单大类列表": {
                "sql": "SELECT DISTINCT flow_category, dept_code FROM dw_flow_category ORDER BY flow_category",
                "connection": "DW",
                "params": [],
            },
            "工单子类列表": {
                "sql": "SELECT DISTINCT flow_type, FLOENTITYDESC, dept_code FROM dw_flow_type ORDER BY flow_type",
                "connection": "DW",
                "params": [],
            },
        }

        override_count = 0
        for name, dw_info in dw_templates.items():
            if name in self._template_store:
                self._template_store[name] = {
                    'sql': dw_info['sql'],
                    'connection': dw_info['connection'],
                    'params': dw_info['params'],
                    'report': 'DW宽表',
                    'dataset': name,
                }
                override_count += 1

        logger.info(f"[AgentQueryEngine] DW宽表模板覆盖: {override_count} 个工单模板已切换到DW连接")

    @staticmethod
    def _fill_template_params(sql: str, params: Dict[str, str]) -> str:
        """
        将模板参数语法替换为实际SQL条件。
        支持两种模板格式：
        1. 帆软${if(...)}语法（原始报表模板）
        2. DW宽表简化占位符：${time_filter}、${dept_filter}、${flow_type_filter}
        """
        import re

        start_time = params.get('start_time', '')
        end_time = params.get('end_time', '')
        dept = params.get('dept', '')
        flow_type = params.get('flow_type', '')
        flow_category = params.get('flow_category', '')

        # ===== 先处理DW宽表简化占位符 =====
        if '${time_filter}' in sql:
            if start_time and end_time:
                time_cond = f"AND ApprovalTime BETWEEN '{start_time}' AND '{end_time} 23:59:59'"
            elif start_time:
                time_cond = f"AND ApprovalTime >= '{start_time}'"
            elif end_time:
                time_cond = f"AND ApprovalTime <= '{end_time} 23:59:59'"
            else:
                time_cond = ""
            sql = sql.replace('${time_filter}', time_cond)

        if '${dept_filter}' in sql:
            if dept:
                dept_values = "','".join(dept.split(','))
                dept_cond = f"AND DEPT_CODE IN ('{dept_values}')"
            else:
                dept_cond = ""
            sql = sql.replace('${dept_filter}', dept_cond)

        if '${flow_type_filter}' in sql:
            if flow_type:
                ft_values = "','".join(flow_type.split(','))
                ft_cond = f"AND FLODESC IN ('{ft_values}')"
            elif flow_category:
                fc_values = "','".join(flow_category.split(','))
                ft_cond = f"AND FLOENTITYDESC IN ('{fc_values}')"
            else:
                ft_cond = ""
            sql = sql.replace('${flow_type_filter}', ft_cond)

        # 如果已无帆软${if(...)}语法，直接返回
        if '${if(' not in sql:
            # 清理残留空行
            sql = re.sub(r'\n\s*\n\s*\n', '\n\n', sql)
            sql = '\n'.join(line.rstrip() for line in sql.split('\n'))
            return sql.strip()

        # ===== 处理帆软${if(...)}语法（原始报表模板） =====
        result = sql

        # 处理时间条件（最复杂的嵌套if）
        if start_time or end_time:
            if start_time and end_time:
                time_cond = f"AND af.LASTSAVED BETWEEN '{start_time}' AND '{end_time} 23:59:59'"
            elif start_time:
                time_cond = f"AND af.LASTSAVED >= '{start_time}'"
            else:
                time_cond = f"AND af.LASTSAVED <= '{end_time} 23:59:59'"
            # 替换包含start_time/end_time的${if()}块
            pattern = r"\$\{if\(len\(start_time\)[^}]*\}\s*"
            result = re.sub(
                r"\$\{if\(len\(start_time\)==0[^}]*\}",
                time_cond,
                result,
                flags=re.DOTALL
            )
            # 更通用的匹配：包含start_time/end_time的整个${if()}块
            def replace_time_if(match):
                return time_cond
            result = re.sub(
                r'\$\{if\(len\(start_time\)[^}]*?\}',
                replace_time_if,
                result,
                flags=re.DOTALL
            )
        else:
            # 没有时间参数，移除整个${if()}时间块
            result = re.sub(
                r'\$\{if\(len\(start_time\)[^}]*?\}',
                '',
                result,
                flags=re.DOTALL
            )

        # 处理dept参数
        if dept:
            dept_values = "','".join(dept.split(','))
            dept_cond = f"AND AU.USRMRC IN ('{dept_values}')"
            # 替换包含dept+USRMRC的${if()}块
            result = re.sub(
                r"\$\{if\(len\(dept\)>0[^}]*?\}",
                dept_cond,
                result,
                flags=re.DOTALL
            )
        else:
            result = re.sub(
                r"\$\{if\(len\(dept\)>0[^}]*?\}",
                '',
                result,
                flags=re.DOTALL
            )

        # 处理flow_type参数
        if flow_type:
            ft_values = "','".join(flow_type.split(','))
            ft_cond = f"AND AF.FLODESC IN ('{ft_values}')"
            result = re.sub(
                r"\$\{if\(len\(flow_type\)>0[^}]*?\}",
                ft_cond,
                result,
                flags=re.DOTALL
            )
        else:
            result = re.sub(
                r"\$\{if\(len\(flow_type\)>0[^}]*?\}",
                '',
                result,
                flags=re.DOTALL
            )

        # 处理flow_category参数
        if flow_category:
            fc_values = "','".join(flow_category.split(','))
            fc_cond = f"AND AF.FLOENTITYDESC IN ('{fc_values}')"
            result = re.sub(
                r"\$\{if\(len\(flow_category\)>0[^}]*?\}",
                fc_cond,
                result,
                flags=re.DOTALL
            )
        else:
            result = re.sub(
                r"\$\{if\(len\(flow_category\)>0[^}]*?\}",
                '',
                result,
                flags=re.DOTALL
            )

        # 处理其他${if(...)}块（如创建人等）- 直接移除
        result = re.sub(r'\$\{if\([^}]*?\}', '', result)

        # 处理简单${param}占位符（如${产品名称}、${采购时间}等）
        lines = result.split('\n')
        filled_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('--'):
                filled_lines.append(line)
            else:
                for key, value in params.items():
                    if value and f'${{{key}}}' in line:
                        line = line.replace(f'${{{key}}}', value)
                filled_lines.append(line)
        result = '\n'.join(filled_lines)

        # 清理残留的空行和多余空格
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        result = '\n'.join(line.rstrip() for line in result.split('\n'))

        return result.strip()

    def _build_system_prompt(self) -> str:
        from datetime import datetime, timedelta
        now = datetime.now()
        first_of_this_month = now.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        # 计算本周一
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        this_week_start = now - timedelta(days=weekday)
        this_week_start = this_week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # 计算上周一和上周日
        last_week_start = this_week_start - timedelta(days=7)
        last_week_end = this_week_start - timedelta(days=1)

        # 最近7天
        recent_7d_start = now - timedelta(days=7)

        # 星期几中文
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

        return AGENT_SYSTEM_PROMPT.format(
            sql_examples=self._sql_examples,
            current_date=now.strftime('%Y-%m-%d'),
            current_year=now.year,
            current_weekday=weekday_names[weekday],
            this_week_start=this_week_start.strftime('%Y-%m-%d'),
            last_week_start=last_week_start.strftime('%Y-%m-%d'),
            last_week_end=last_week_end.strftime('%Y-%m-%d'),
            this_month_start=first_of_this_month.strftime('%Y-%m-%d'),
            last_month_start=last_month_start.strftime('%Y-%m-%d'),
            last_month_end=last_month_end.strftime('%Y-%m-%d'),
            recent_7d_start=recent_7d_start.strftime('%Y-%m-%d'),
            example_start=first_of_this_month.strftime('%Y-%m-%d'),
            example_end=now.strftime('%Y-%m-%d'),
        )

    def _execute_sql(self, sql: str, connection_name: str) -> Dict:
        """执行SQL查询，委托给统一的 db_executor"""
        return execute_query(sql, connection_name=connection_name, max_rows=MAX_SQL_ROWS)

    def _normalize_sql(self, sql: str, connection_name: str = "EAM") -> str:
        import re

        # DW是MySQL，不需要SQL Server语法转换
        is_mysql = connection_name.upper() == "DW"

        if is_mysql:
            # MySQL: TOP N → LIMIT N (LLM可能仍生成SQL Server语法)
            top_match = re.search(r'\bSELECT\s+TOP\s+(\d+)\s+', sql, re.IGNORECASE)
            if top_match:
                n = top_match.group(1)
                sql = re.sub(r'\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', sql, flags=re.IGNORECASE)
                sql = sql.rstrip().rstrip(';')
                if not re.search(r'\bLIMIT\s+', sql, re.IGNORECASE):
                    sql = sql + f'\nLIMIT {n}'
                logger.info(f"[AgentQueryEngine] 自动修复: TOP {n} → LIMIT {n} (MySQL)")
            # MySQL: CONVERT(varchar(...), field, 120) → DATE_FORMAT(field, '%Y-%m-%d')
            convert_match = re.search(r"CONVERT\s*\(\s*varchar\s*\(\s*\d+\s*\)\s*,\s*([\w.]+)\s*,\s*120\s*\)", sql, re.IGNORECASE)
            if convert_match:
                field = convert_match.group(1)
                sql = re.sub(
                    r"CONVERT\s*\(\s*varchar\s*\(\s*\d+\s*\)\s*,\s*[\w.]+\s*,\s*120\s*\)",
                    f"DATE_FORMAT({field}, '%Y-%m-%d')",
                    sql,
                    flags=re.IGNORECASE
                )
                logger.info(f"[AgentQueryEngine] 自动修复: CONVERT → DATE_FORMAT (MySQL)")
            # MySQL: GETDATE() → NOW()
            sql = re.sub(r'\bGETDATE\s*\(\s*\)', 'NOW()', sql, flags=re.IGNORECASE)
            # MySQL: DATEADD → DATE_ADD / DATE_SUB
            dateadd_match = re.search(r"DATEADD\s*\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*([\w.()]+)\s*\)", sql, re.IGNORECASE)
            if dateadd_match:
                unit, num, base = dateadd_match.group(1), dateadd_match.group(2), dateadd_match.group(3)
                n = int(num)
                unit_map = {'DAY': 'DAY', 'HOUR': 'HOUR', 'WEEK': 'WEEK', 'MONTH': 'MONTH', 'YEAR': 'YEAR', 'MINUTE': 'MINUTE'}
                mysql_unit = unit_map.get(unit.upper(), 'DAY')
                if n >= 0:
                    sql = re.sub(r"DATEADD\s*\([^)]+\)", f"DATE_ADD({base}, INTERVAL {n} {mysql_unit})", sql, count=1, flags=re.IGNORECASE)
                else:
                    sql = re.sub(r"DATEADD\s*\([^)]+\)", f"DATE_SUB({base}, INTERVAL {abs(n)} {mysql_unit})", sql, count=1, flags=re.IGNORECASE)
                logger.info(f"[AgentQueryEngine] 自动修复: DATEADD → DATE_ADD/DATE_SUB (MySQL)")
            # MySQL: DATEDIFF → TIMESTAMPDIFF
            datediff_match = re.search(r"DATEDIFF\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", sql, re.IGNORECASE)
            if datediff_match:
                unit, start, end = datediff_match.group(1), datediff_match.group(2).strip(), datediff_match.group(3).strip()
                unit_map = {'DAY': 'DAY', 'HOUR': 'HOUR', 'MINUTE': 'MINUTE', 'SECOND': 'SECOND'}
                mysql_unit = unit_map.get(unit.upper(), 'DAY')
                sql = re.sub(r"DATEDIFF\s*\([^)]+\)", f"TIMESTAMPDIFF({mysql_unit}, {start}, {end})", sql, count=1, flags=re.IGNORECASE)
                logger.info(f"[AgentQueryEngine] 自动修复: DATEDIFF → TIMESTAMPDIFF (MySQL)")
        else:
            # SQL Server: LIMIT → TOP
            if re.search(r'\bLIMIT\s+\d+', sql, re.IGNORECASE):
                limit_match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
                if limit_match:
                    n = limit_match.group(1)
                    sql = re.sub(r'\bLIMIT\s+\d+', '', sql, flags=re.IGNORECASE).rstrip().rstrip(';')
                    if not re.search(r'\bTOP\s+', sql, re.IGNORECASE):
                        sql = sql.replace('SELECT', f'SELECT TOP {n}', 1)
                    logger.info(f"[AgentQueryEngine] 自动修复: LIMIT {n} → TOP {n}")

        # 字符串拼接: col + '|' + col → CONCAT(col, '|', col)
        sql = re.sub(
            r"([\w.]+)\s*\+\s*'([^']*)'\s*\+\s*([\w.]+)",
            r"CONCAT(\1, '\2', \3)",
            sql
        )

        # CTE查询修正（仅SQL Server）
        if not is_mysql and 'ProcessTime' in sql and 'WITH Ordered' in sql:
            logger.info("[AgentQueryEngine] 检测到CTE查询，自动修正为标准模板")
            sql = self._fix_cte_sql(sql)

        return sql

    def _fix_cte_sql(self, sql: str) -> str:
        import re
        dept_filter = ''
        dept_match = re.search(r"USRMRC\s*=\s*'(\w+)'", sql)
        if dept_match:
            dept_filter = f"  AND AU.USRMRC = '{dept_match.group(1)}'"

        time_filter = ''
        time_match = re.search(r"LASTSAVED\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'", sql)
        if not time_match:
            time_match = re.search(r"ApprovalTime\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'", sql)
        if not time_match:
            time_match = re.search(r"CreateTime\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'", sql)
        if time_match:
            time_filter = f"  AND da.ApprovalTime BETWEEN '{time_match.group(1)}' AND '{time_match.group(2)}'"

        has_dept_name = 'USRMRCDESC' in sql or '部门名称' in sql
        dept_name_select = ', AU.USRMRCDESC AS 部门名称' if has_dept_name else ''
        dept_name_group = ', AU.USRMRCDESC' if has_dept_name else ''

        fixed = f"""WITH Ordered AS (
    SELECT RECCODE, RECFLOCODE, CREATEDBY, LASTSAVED,
      ROW_NUMBER() OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY LASTSAVED, RECCODE) AS NewNode
    FROM ATWORKFLOWRECORDS
),
ProcessTime AS (
    SELECT RECCODE, RECFLOCODE, CREATEDBY,
      LASTSAVED AS ApprovalTime,
      LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode) AS CreateTime,
      CASE WHEN LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode) IS NOT NULL
        THEN ROUND(CAST(DATEDIFF(MINUTE, LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode), LASTSAVED) AS FLOAT) / 1440.0, 2)
        ELSE NULL END AS ProcessDays
    FROM Ordered
),
data_all AS (
    SELECT RECCODE, RECFLOCODE, CREATEDBY, CreateTime, ApprovalTime, ProcessDays
    FROM ProcessTime WHERE CreateTime IS NOT NULL AND ProcessDays > 0
)
SELECT AU.USRMRC AS 部门代码{dept_name_select},
       COUNT(DISTINCT CONCAT(da.RECCODE, '|', da.RECFLOCODE)) AS 工单数量,
       ROUND(SUM(da.ProcessDays), 3) AS 总处理天数,
       ROUND(AVG(da.ProcessDays), 3) AS 平均处理天数,
       COUNT(DISTINCT da.CREATEDBY) AS 处理人数
FROM data_all da
LEFT JOIN ATUSERS AU ON da.CREATEDBY = AU.USRCODE
LEFT JOIN ATWORKFLOW AF ON da.RECFLOCODE = AF.FLOCODE
WHERE AU.USRDESC IS NOT NULL AND AU.USRMRC IS NOT NULL AND AU.USRMRC <> '*'{dept_filter}{time_filter}
GROUP BY AU.USRMRC{dept_name_group}
ORDER BY 平均处理天数 ASC"""
        return fixed

    def _handle_tool_call(self, tool_name: str, tool_args: Dict, step: int) -> Generator:
        if tool_name == "query_by_template":
            # 模板查询：使用帆软报表原始SQL模板，保证数据一致性
            template_name = tool_args.get("template_name", "")
            template_params = {}
            for key in ["start_time", "end_time", "dept", "flow_type", "flow_category"]:
                val = tool_args.get(key, "")
                if val:
                    template_params[key] = val

            # ⚠️ 自动补全：如果用户有时间意图但LLM没传时间参数，自动补充
            auto_time_params = self._extract_time_params(self._current_user_query or "")
            if "start_time" not in template_params and "start_time" in auto_time_params:
                template_params["start_time"] = auto_time_params["start_time"]
                logger.info(f"[模板参数补全] 自动补充start_time={auto_time_params['start_time']}")
            if "end_time" not in template_params and "end_time" in auto_time_params:
                template_params["end_time"] = auto_time_params["end_time"]
                logger.info(f"[模板参数补全] 自动补充end_time={auto_time_params['end_time']}")
            # ⚠️ 自动补全：如果用户提到部门但LLM没传dept参数
            if "dept" not in template_params and "dept" in auto_time_params:
                template_params["dept"] = auto_time_params["dept"]
                logger.info(f"[模板参数补全] 自动补充dept={auto_time_params['dept']}")

            yield {"type": "thinking", "content": f"使用模板查询: {template_name}"}

            # 查找模板
            template_info = self._template_store.get(template_name)
            if not template_info:
                error_msg = f"未找到模板: {template_name}，可用模板: {list(self._template_store.keys())}"
                logger.warning(f"[AgentQueryEngine] {error_msg}")
                yield {"type": "data", "data": [], "step": step, "error": error_msg}
                return error_msg

            raw_sql = template_info['sql']
            connection_name = template_info['connection']

            # 填充参数：将帆软${if(...)}语法替换为实际SQL条件
            filled_sql = self._fill_template_params(raw_sql, template_params)

            # ⚠️ 模板SQL后处理：员工分组修正（与execute_sql路径一致）
            filled_sql = self._fix_group_by_for_employee(filled_sql, self._current_user_query or "")
            # ⚠️ 模板SQL后处理：计数逻辑修正
            filled_sql = self._fix_count_logic(filled_sql)
            # ⚠️ 模板SQL后处理：SQL语法修正（DW连接时TOP→LIMIT等）
            filled_sql = self._normalize_sql(filled_sql, connection_name)

            logger.info(
                f"[AgentQueryEngine] 模板查询 | 模板={template_name} | "
                f"连接={connection_name} | 参数={template_params}"
            )
            logger.debug(f"[AgentQueryEngine] 填充后SQL:\n{filled_sql[:500]}")

            yield {"type": "sql", "sql": filled_sql, "step": step, "template": template_name}

            # 执行SQL（直连数据库）
            result = self._execute_sql(filled_sql, connection_name)

            if result.get("success"):
                data = result["data"]
                yield {
                    "type": "data",
                    "data": data,
                    "step": step,
                    "total_rows": result.get("total_rows", len(data)),
                    "truncated": result.get("truncated", False),
                    "elapsed": result.get("elapsed", 0),
                    "template": template_name,
                }
                # 【修复图表消失】query_by_template路径也需要自动生成图表
                if data:
                    auto_chart = self._auto_generate_chart(data, self._current_user_query or "")
                    if auto_chart:
                        yield auto_chart
                    # 工单时序数据深度分析
                    ts_event = self._try_timeseries_analysis(data, self._current_user_query or "")
                    if ts_event:
                        yield ts_event
                data_summary = safe_json_dumps(data[:50], ensure_ascii=False)
                if len(data) > 50:
                    data_summary += f"\n... (共{len(data)}条，仅展示前50条)"
                return data_summary
            else:
                error_msg = result.get("error", "未知错误")
                yield {"type": "data", "data": [], "step": step, "error": error_msg}
                return f"模板SQL执行失败: {error_msg}"

        elif tool_name == "execute_sql":
            sql = tool_args.get("sql", "")
            connection_name = tool_args.get("connection_name", "DW")

            # 时间范围校验与修正
            sql = self._fix_time_range(sql, self._current_user_query, connection_name=connection_name)
            # 工单计数逻辑校验与修正
            sql = self._fix_count_logic(sql)
            # 员工分组校验与修正
            sql = self._fix_group_by_for_employee(sql, self._current_user_query, connection_name=connection_name)

            yield {"type": "sql", "sql": sql, "step": step}

            sql = self._normalize_sql(sql, connection_name)

            level, message, details = self.safety_validator.validate(sql)
            if level == SecurityLevel.DANGEROUS:
                error_msg = f"SQL安全验证未通过: {message}"
                logger.warning(f"[AgentQueryEngine] {error_msg}")
                yield {"type": "data", "data": [], "step": step, "error": error_msg}
                return error_msg
            elif level == SecurityLevel.WARNING:
                logger.warning(f"[AgentQueryEngine] SQL警告(允许执行): {message}")

            result = self._execute_sql(sql, connection_name)

            if result.get("success"):
                data = result["data"]
                yield {
                    "type": "data",
                    "data": data,
                    "step": step,
                    "total_rows": result.get("total_rows", len(data)),
                    "truncated": result.get("truncated", False),
                    "elapsed": result.get("elapsed", 0)
                }
                data_summary = safe_json_dumps(data[:50], ensure_ascii=False)
                if len(data) > 50:
                    data_summary += f"\n... (共{len(data)}条，仅展示前50条)"
                return data_summary
            else:
                error_msg = result.get("error", "未知错误")
                yield {"type": "data", "data": [], "step": step, "error": error_msg}
                return f"SQL执行失败: {error_msg}"

        elif tool_name == "query_scada":
            devices_str = tool_args.get("devices", "")
            time_range_str = tool_args.get("time_range", "最近1天")
            analysis_type = tool_args.get("analysis_type", "raw")
            threshold = tool_args.get("threshold")
            threshold_operator = tool_args.get("threshold_operator", ">")

            yield {"type": "thinking", "content": f"正在查询SCADA数据: {devices_str}"}

            try:
                from services.scada_analyzer import SCADAAnalyzer
                from utils.db_config import HISTORIAN_CONFIG

                analyzer = SCADAAnalyzer(HISTORIAN_CONFIG)

                # 解析查询（拼接完整信息，确保 parse_scada_query 能提取阈值等）
                query_parts = [devices_str, time_range_str]
                if threshold is not None:
                    op_text = "超过" if threshold_operator == ">" else "低于"
                    query_parts.append(f"{op_text}{threshold}度")
                full_query = " ".join(query_parts)
                parsed = analyzer.parse_scada_query(full_query)

                if not parsed["tagnames"]:
                    return f"未找到匹配的SCADA设备: {devices_str}"

                tagnames = parsed["tagnames"]
                device_info = parsed["device_info"]
                time_range = parsed["time_range"]

                yield {"type": "sql", "sql": f"-- SCADA查询: {', '.join(tagnames)} | {time_range.get('start_date')} ~ {time_range.get('end_date')}", "step": step}

                # 获取时序数据
                multi_data = analyzer.fetch_timeseries(
                    tagnames=tagnames,
                    start_date=time_range["start_date"],
                    end_date=time_range["end_date"],
                    resolution=parsed.get("resolution", 60000),
                )

                if not any(multi_data.values()):
                    return f"SCADA数据查询失败: 未获取到任何数据，请检查Historian数据库连接"

                # 根据分析类型处理
                analysis_result = {}
                chart_config = None

                if analysis_type == "threshold" and threshold is not None:
                    # 阈值分析 - 对第一个设备做阈值分析
                    first_tagname = tagnames[0]
                    first_data = multi_data.get(first_tagname, [])
                    analysis_result = analyzer.analyze_threshold(
                        first_data, threshold, threshold_operator
                    )
                    chart_config = analyzer.build_chart_config(
                        multi_data, device_info, "threshold",
                        {"threshold": threshold, "operator": threshold_operator}
                    )
                    # 构建返回摘要
                    total_min = analysis_result.get("total_duration_min", 0)
                    exceed_count = analysis_result.get("exceed_count", 0)
                    op_desc = "超过" if threshold_operator == ">" else "低于"
                    summary = (
                        f"在查询时间段内，{device_info.get(first_tagname, {}).get('cn_desc', first_tagname)}"
                        f" {op_desc} {threshold}的次数: {exceed_count}次，"
                        f"累计持续: {total_min}分钟"
                    )
                    if analysis_result.get("periods"):
                        longest = max(analysis_result["periods"], key=lambda p: p["duration_min"])
                        summary += f"，最长持续: {longest['duration_min']}分钟({longest['start']}~{longest['end']})"

                    yield {
                        "type": "scada_analysis",
                        "analysis_type": "threshold",
                        "analysis_result": analysis_result,
                        "device_info": device_info,
                        "chart_config": chart_config,
                        "step": step,
                        "summary": summary,
                    }
                    return summary

                elif analysis_type == "comparison" or len(tagnames) > 1:
                    # 对比分析
                    analysis_result = analyzer.analyze_comparison(multi_data, device_info)
                    chart_config = analyzer.build_chart_config(multi_data, device_info, "comparison")

                    # 构建对比摘要
                    comparison_lines = []
                    for tagname, stats in analysis_result.get("devices", {}).items():
                        comparison_lines.append(
                            f"{stats.get('label', tagname)}: "
                            f"当前{stats.get('current', 'N/A')}{stats.get('unit', '')}, "
                            f"最高{stats.get('max', 'N/A')}{stats.get('unit', '')}, "
                            f"最低{stats.get('min', 'N/A')}{stats.get('unit', '')}, "
                            f"平均{stats.get('avg', 'N/A')}{stats.get('unit', '')}"
                        )
                    summary = "\n".join(comparison_lines)

                    yield {
                        "type": "scada_analysis",
                        "analysis_type": "comparison",
                        "analysis_result": analysis_result,
                        "device_info": device_info,
                        "chart_config": chart_config,
                        "step": step,
                        "summary": summary,
                    }
                    return summary

                elif analysis_type == "trend":
                    # 趋势分析
                    first_tagname = tagnames[0]
                    first_data = multi_data.get(first_tagname, [])
                    analysis_result = analyzer.analyze_trend(first_data)
                    chart_config = analyzer.build_chart_config(multi_data, device_info, "trend")

                    trend_desc = {"rising": "上升", "declining": "下降", "stable": "稳定"}
                    summary = (
                        f"{device_info.get(first_tagname, {}).get('cn_desc', first_tagname)}: "
                        f"趋势{trend_desc.get(analysis_result.get('trend', ''), '未知')}, "
                        f"总变化{analysis_result.get('total_change', 0):.2f}, "
                        f"每小时变化{analysis_result.get('rate_per_hour', 0):.4f}"
                    )

                    yield {
                        "type": "scada_analysis",
                        "analysis_type": "trend",
                        "analysis_result": analysis_result,
                        "device_info": device_info,
                        "chart_config": chart_config,
                        "step": step,
                        "summary": summary,
                    }
                    return summary

                else:
                    # 原始数据展示
                    chart_config = analyzer.build_chart_config(multi_data, device_info)

                    # 统计摘要
                    stats_lines = []
                    for tagname, data in multi_data.items():
                        if not data:
                            continue
                        values = [p["value"] for p in data]
                        info = device_info.get(tagname, {})
                        stats_lines.append(
                            f"{info.get('cn_desc', tagname)}: "
                            f"当前{values[-1]:.2f}{info.get('unit', '')}, "
                            f"最高{max(values):.2f}, 最低{min(values):.2f}, "
                            f"平均{sum(values)/len(values):.2f}, 共{len(values)}个数据点"
                        )
                    summary = "\n".join(stats_lines)

                    yield {
                        "type": "scada_analysis",
                        "analysis_type": "raw",
                        "analysis_result": {"devices": {
                            tn: {"count": len(d), "data": d[:5]}
                            for tn, d in multi_data.items()
                        }},
                        "device_info": device_info,
                        "chart_config": chart_config,
                        "step": step,
                        "summary": summary,
                    }
                    return summary

            except Exception as e:
                logger.error(f"[AgentQueryEngine] SCADA查询失败: {e}")
                return f"SCADA查询失败: {str(e)}"

        return f"未知工具: {tool_name}"

    def _build_echarts_config(self, data: List[Dict], chart_type: str, title: str) -> Dict:
        if not data:
            return {"title": {"text": title or "无数据"}, "series": []}

        columns = list(data[0].keys()) if data else []
        x_field = columns[0] if columns else ""
        y_fields = [c for c in columns[1:] if any(isinstance(row.get(c), (int, float)) for row in data)]

        if not y_fields:
            y_fields = columns[1:2] if len(columns) > 1 else []

        x_data = [str(row.get(x_field, "")) for row in data]

        series = []
        for yf in y_fields:
            y_data = [row.get(yf, 0) for row in data]
            series.append({
                "name": yf,
                "type": chart_type,
                "data": y_data
            })

        # 截断过长的x轴标签
        x_data_display = [str(x)[:8] + '..' if len(str(x)) > 8 else str(x) for x in x_data]

        config = {
            "title": {"text": title, "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis" if chart_type in ("bar", "line") else "item"},
            "legend": {"data": y_fields, "top": 25},
            "grid": {"left": "3%", "right": "4%", "bottom": "12%", "top": "18%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": x_data_display,
                "axisLabel": {
                    "rotate": 30 if any(len(str(x)) > 4 for x in x_data) else 0,
                    "interval": 0,
                    "fontSize": 11,
                }
            } if chart_type in ("bar", "line") else {},
            "yAxis": {"type": "value"} if chart_type in ("bar", "line") else {},
            "series": series
        }

        if chart_type == "pie" and y_fields:
            pie_data = []
            for row in data:
                pie_data.append({
                    "name": str(row.get(x_field, "")),
                    "value": row.get(y_fields[0], 0)
                })
            config["series"] = [{
                "name": y_fields[0],
                "type": "pie",
                "radius": "60%",
                "data": pie_data
            }]
            del config["xAxis"]
            del config["yAxis"]

        return config

    def _auto_generate_chart(self, data: List[Dict], user_query: str) -> Optional[Dict]:
        """
        根据查询数据自动生成合适的ECharts图表配置
        - 有1个分类维度+1-3个数值列 → 柱状图
        - 有1个分类维度+1个数值列且分类数<=8 → 饼图
        - 有时间维度+1-3个数值列 → 折线图
        - 工单时序数据 → 额外调用TimeSeriesAnalyzer做ADTK深度分析
        """
        if not data or len(data) < 2:
            return None

        columns = list(data[0].keys())
        if len(columns) < 2:
            return None

        # 识别分类列和数值列
        numeric_cols = []
        category_cols = []
        for col in columns:
            numeric_count = sum(1 for row in data if isinstance(row.get(col), (int, float)) and row.get(col) is not None)
            if numeric_count > len(data) * 0.5:
                numeric_cols.append(col)
            else:
                category_cols.append(col)

        if not numeric_cols or not category_cols:
            return None

        x_field = category_cols[0]
        y_fields = numeric_cols[:3]  # 最多3个数值列

        # 判断是否是时间序列
        is_time_series = any(kw in user_query for kw in ['趋势', '曲线', '变化', '走势', '时间'])
        time_keywords = ['时间', '日期', 'date', 'time', '月', '天', '周', '年']
        if any(kw in x_field.lower() for kw in time_keywords):
            is_time_series = True

        # 选择图表类型
        if is_time_series:
            chart_type = "line"
        elif len(data) <= 8 and len(y_fields) == 1:
            chart_type = "pie"
        else:
            chart_type = "bar"

        # 构建ECharts配置
        x_data = [str(row.get(x_field, "")) for row in data]

        if chart_type == "pie":
            pie_data = []
            for row in data:
                pie_data.append({
                    "name": str(row.get(x_field, "")),
                    "value": row.get(y_fields[0], 0)
                })
            config = {
                "title": {"text": user_query[:20], "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "legend": {"orient": "vertical", "left": "left", "top": "middle"},
                "series": [{
                    "name": y_fields[0],
                    "type": "pie",
                    "radius": ["30%", "60%"],
                    "data": pie_data,
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}},
                    "label": {"formatter": "{b}\n{d}%"}
                }]
            }
        else:
            series = []
            colors = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"]
            for i, yf in enumerate(y_fields):
                y_data = [row.get(yf, 0) for row in data]
                s = {
                    "name": yf,
                    "type": chart_type,
                    "data": y_data,
                }
                if chart_type == "bar":
                    s["itemStyle"] = {"color": colors[i % len(colors)]}
                    s["barMaxWidth"] = 40
                elif chart_type == "line":
                    s["smooth"] = True
                    s["symbol"] = "circle"
                    s["symbolSize"] = 6
                series.append(s)

            config = {
                "title": {"text": user_query[:20], "left": "center"},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": y_fields, "top": 30},
                "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
                "xAxis": {
                    "type": "category",
                    "data": x_data,
                    "axisLabel": {"rotate": len(str(x_data[0])) > 4 and 30 or 0, "interval": 0}
                },
                "yAxis": {"type": "value"},
                "series": series
            }

        return {"type": "chart", "config": config}

    def _try_timeseries_analysis(self, data: List[Dict], user_query: str) -> Optional[Dict]:
        """
        尝试对工单等SQL查询结果做时序深度分析（基于ADTK）

        当数据满足以下条件时触发：
        1. 数据行数 >= 5（足够做时序分析）
        2. 存在可识别的时间列和数值列
        3. 用户查询包含时序分析意图（趋势/异常/阈值/对比/聚合）

        返回: workorder_analysis 事件 Dict 或 None
        """
        if not data or len(data) < 5:
            return None

        # 检测用户是否有时序分析意图
        analysis_keywords = [
            '趋势', '变化', '走势', '波动', '上升', '下降',
            '异常', '突变', '阶跃', '不正常',
            '超过', '大于', '高于', '低于', '小于', '超标', '超限',
            '对比', '比较', 'vs',
            '平均', '均值', '每天', '每小时', '日均', '统计', '汇总',
            '持续', '时长', '耗时', '效率', '效能',
        ]
        has_ts_intent = any(kw in user_query for kw in analysis_keywords)
        if not has_ts_intent:
            return None

        # 识别时间列和数值列
        columns = list(data[0].keys())
        time_col = None
        numeric_cols = []

        # 时间列识别
        time_keywords_in_col = ['时间', '日期', 'date', 'time', 'ApprovalTime', 'CreateTime',
                                'LASTSAVED', 'hour', 'day', '月', '周', '年']
        for col in columns:
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in time_keywords_in_col):
                # 验证该列确实包含时间值
                sample_val = str(data[0].get(col, ""))
                if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}', sample_val):
                    time_col = col
                    break

        if not time_col:
            return None

        # 数值列识别
        for col in columns:
            if col == time_col:
                continue
            numeric_count = sum(1 for row in data
                              if isinstance(row.get(col), (int, float)) and row.get(col) is not None)
            if numeric_count > len(data) * 0.5:
                numeric_cols.append(col)

        if not numeric_cols:
            return None

        # 将SQL行数据转为时序格式 [{datetime, value}, ...]
        try:
            from services.scada_analyzer import TimeSeriesAnalyzer

            analyzer = TimeSeriesAnalyzer()
            primary_col = numeric_cols[0]

            # 转换数据
            ts_data = []
            for row in data:
                dt_val = row.get(time_col)
                num_val = row.get(primary_col)
                if dt_val is None or num_val is None:
                    continue
                # 格式化datetime
                dt_str = str(dt_val)
                if isinstance(dt_val, datetime):
                    dt_str = dt_val.strftime("%Y-%m-%d %H:%M:%S")
                # 截断小数秒
                dt_str = dt_str.split('.')[0]
                try:
                    ts_data.append({"datetime": dt_str, "value": float(num_val)})
                except (ValueError, TypeError):
                    continue

            if len(ts_data) < 5:
                return None

            # 判断分析类型
            analysis_type = "raw"
            threshold = None
            threshold_operator = ">"

            # 阈值意图
            threshold_patterns = [
                (r'超过\s*(\d+\.?\d*)', '>'),
                (r'大于\s*(\d+\.?\d*)', '>'),
                (r'高于\s*(\d+\.?\d*)', '>'),
                (r'低于\s*(\d+\.?\d*)', '<'),
                (r'小于\s*(\d+\.?\d*)', '<'),
            ]
            for pattern, op in threshold_patterns:
                m = re.search(pattern, user_query)
                if m:
                    threshold = float(m.group(1))
                    threshold_operator = op
                    analysis_type = "threshold"
                    break

            if analysis_type == "raw":
                if re.search(r'异常|突变|阶跃|波动异常|不正常', user_query):
                    analysis_type = "anomaly"
                elif re.search(r'对比|比较|vs', user_query):
                    analysis_type = "comparison"
                elif re.search(r'趋势|变化|走势|波动', user_query):
                    analysis_type = "trend"
                elif re.search(r'平均|均值|每天|每小时|日均|统计|汇总', user_query):
                    analysis_type = "aggregation"

            # 执行分析
            analysis_result = {}
            chart_config = None
            summary = ""

            # 构建设备信息（用于图表）
            col_label = primary_col
            # 中文列名映射
            col_name_map = {
                'order_count': '工单数量', 'avg_process_days': '平均处理天数',
                'total_process_days': '总处理天数', 'ProcessHours': '处理时长(小时)',
                'ProcessDays': '处理时长(天)', 'user_count': '处理人数',
                '工单数量': '工单数量', '审批数量': '审批数量',
                '总耗时': '总耗时', 'avg_process_hours': '平均处理时长(小时)',
            }
            col_label = col_name_map.get(primary_col, primary_col)

            device_info = {
                primary_col: {
                    "cn_desc": col_label,
                    "room_name": "",
                    "measure_label": col_label,
                    "unit": "",
                    "color": "#4fc3f7",
                }
            }
            multi_data = {primary_col: ts_data}

            if analysis_type == "threshold" and threshold is not None:
                analysis_result = analyzer.analyze_threshold(ts_data, threshold, threshold_operator)
                chart_config = analyzer.build_chart_config(
                    multi_data, device_info, "threshold",
                    {"threshold": threshold, "operator": threshold_operator}
                )
                op_desc = "超过" if threshold_operator == ">" else "低于"
                summary = f"{col_label} {op_desc}{threshold}的次数: {analysis_result.get('exceed_count', 0)}次，累计持续: {analysis_result.get('total_duration_min', 0)}分钟"

            elif analysis_type == "anomaly":
                analysis_result = analyzer.analyze_anomaly(ts_data)
                chart_config = analyzer.build_chart_config(
                    multi_data, device_info, "anomaly",
                    anomaly_data={primary_col: analysis_result}
                )
                summary = f"检测到{analysis_result.get('total_anomaly_events', 0)}个异常事件，检测器: {', '.join(analysis_result.get('detectors_used', []))}"

            elif analysis_type == "trend":
                analysis_result = analyzer.analyze_trend(ts_data)
                chart_config = analyzer.build_chart_config(multi_data, device_info, "trend")
                trend_desc = {"rising": "上升", "declining": "下降", "stable": "稳定"}
                summary = f"{col_label}趋势: {trend_desc.get(analysis_result.get('trend', ''), '未知')}，总变化: {analysis_result.get('total_change', 0):.2f}"

            elif analysis_type == "aggregation":
                interval = "day" if len(ts_data) > 48 else "hour"
                agg_result = analyzer.analyze_aggregation(ts_data, interval=interval)
                analysis_result = agg_result
                chart_config = analyzer.build_chart_config(
                    multi_data, device_info, "aggregation",
                    aggregation_data={primary_col: agg_result}
                )
                summary = f"{col_label}聚合统计: {len(agg_result)}个时段"

            elif analysis_type == "comparison":
                # 对比分析需要多列数据，用前两个数值列
                if len(numeric_cols) >= 2:
                    second_col = numeric_cols[1]
                    second_label = col_name_map.get(second_col, second_col)
                    ts_data_2 = []
                    for row in data:
                        dt_val = row.get(time_col)
                        num_val = row.get(second_col)
                        if dt_val is None or num_val is None:
                            continue
                        dt_str = str(dt_val).split('.')[0]
                        try:
                            ts_data_2.append({"datetime": dt_str, "value": float(num_val)})
                        except (ValueError, TypeError):
                            continue
                    multi_data = {primary_col: ts_data, second_col: ts_data_2}
                    device_info[second_col] = {
                        "cn_desc": second_label, "room_name": "",
                        "measure_label": second_label, "unit": "", "color": "#66bb6a",
                    }
                    analysis_result = analyzer.analyze_comparison(multi_data, device_info)
                    chart_config = analyzer.build_chart_config(multi_data, device_info, "comparison")
                    summary = f"{col_label} vs {second_label} 对比分析"
                else:
                    # 单列无法对比，降级为趋势
                    analysis_result = analyzer.analyze_trend(ts_data)
                    chart_config = analyzer.build_chart_config(multi_data, device_info, "trend")
                    analysis_type = "trend"
                    summary = f"{col_label}趋势分析"

            else:
                # raw: 基础统计
                analysis_result = {"count": len(ts_data)}
                chart_config = analyzer.build_chart_config(multi_data, device_info, "raw")
                summary = f"{col_label}时序数据，共{len(ts_data)}个数据点"

            logger.info(f"[AgentQueryEngine] 工单时序分析: type={analysis_type}, col={primary_col}, points={len(ts_data)}")

            return {
                "type": "workorder_analysis",
                "analysis_type": analysis_type,
                "analysis_result": analysis_result,
                "device_info": device_info,
                "chart_config": chart_config,
                "primary_col": primary_col,
                "col_label": col_label,
                "time_col": time_col,
                "summary": summary,
            }

        except Exception as e:
            logger.warning(f"[AgentQueryEngine] 工单时序分析失败: {e}")
            return None

    def _generate_summary(self, user_query: str, data: List[Dict]) -> Generator:
        """基于查询数据生成流式总结（只调用1次LLM）"""
        if not data:
            yield {"type": "text", "content": "查询返回0条数据，无法生成分析。请尝试调整查询条件。"}
            yield {"type": "done", "summary": "无数据"}
            return

        rows = data[:20]
        data_str = safe_json_dumps(rows, ensure_ascii=False)

        summary_prompt = (
            f"用户问题: {user_query}\n\n"
            f"以下是查询获取到的数据结果:\n{data_str}\n\n"
            f"请基于以上数据结果，用简洁的中文直接回答用户的问题。"
            f"必须使用数据中的具体数字，不要用XX代替。"
        )

        stream_messages = [
            {"role": "system", "content": "你是数据分析助手，请基于查询结果数据回答问题，必须使用具体数字。"},
            {"role": "user", "content": summary_prompt},
        ]

        full_text = ""
        for chunk in self.llm_service.chat_stream(stream_messages, max_tokens=800, temperature=0.1):
            if chunk and not chunk.startswith("[错误]"):
                full_text += chunk
                yield {"type": "text", "content": chunk}

        yield {"type": "done", "summary": full_text[:200]}

    # 本地模板路由关键词映射（在LLM之前匹配，跳过推理轮）
    _LOCAL_ROUTE_PATTERNS = [
        # (关键词列表, 模板名, 连接) — 仅保留精确匹配场景，其他走LLM推理
        # DW宽表SQL足够简单，LLM可以直接生成，不需要模板路由
        (["处理时长分布", "耗时分布", "3小时内", "1天到5天"], "处理时长分布", "DW"),
        (["待审批", "审批停留", "待处理"], "待审批工单", "DW"),
        # WMS相关（暂未建DW宽表，仍用原始库）
        (["库区库存", "库存汇总", "各库区"], "各库区库存汇总", "WMS_PROD"),
        (["库存总数"], "库存总数", "WMS_PROD"),
        (["呆滞物料", "呆滞"], "呆滞物料", "WMS_PROD"),
        (["近效期", "即将过期", "效期物料"], "近效期物料", "WMS_PROD"),
    ]

    def _try_local_route(self, user_query: str) -> Optional[Dict]:
        """
        本地模板路由：关键词匹配模板，跳过LLM推理轮
        返回 None 表示未匹配，需要走LLM推理
        返回 Dict 表示匹配成功，包含 template_name 和 connection
        """
        q = user_query.lower()
        best_match = None
        best_score = 0

        for keywords, template_name, connection in self._LOCAL_ROUTE_PATTERNS:
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best_match = {
                    "template_name": template_name,
                    "connection": connection,
                    "score": score,
                }

        # 至少匹配1个关键词且模板存在
        if best_match and best_match["score"] >= 1 and best_match["template_name"] in self._template_store:
            return best_match
        return None

    def _extract_time_params(self, user_query: str) -> Dict[str, str]:
        """从用户查询中提取时间参数"""
        from datetime import datetime, timedelta
        import re

        params = {}
        now = datetime.now()
        q = user_query

        # 本周
        if "本周" in q:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            params["start_time"] = this_week_start.strftime('%Y-%m-%d')
            params["end_time"] = now.strftime('%Y-%m-%d')
        # 上周
        elif "上周" in q:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(days=1)
            params["start_time"] = last_week_start.strftime('%Y-%m-%d')
            params["end_time"] = last_week_end.strftime('%Y-%m-%d')
        # 本月
        elif "本月" in q:
            params["start_time"] = now.replace(day=1).strftime('%Y-%m-%d')
            params["end_time"] = now.strftime('%Y-%m-%d')
        # 上月
        elif "上月" in q:
            first_of_this_month = now.replace(day=1)
            last_month_end = first_of_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            params["start_time"] = last_month_start.strftime('%Y-%m-%d')
            params["end_time"] = last_month_end.strftime('%Y-%m-%d')
        # 最近N天
        m = re.search(r'最近(\d+)天', q)
        if m:
            days = int(m.group(1))
            params["start_time"] = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            params["end_time"] = now.strftime('%Y-%m-%d')
        # 最近N个月
        elif re.search(r'最近(\d+)(?:个)?月', q):
            m = re.search(r'最近(\d+)(?:个)?月', q)
            months = int(m.group(1))
            month_ago = now - timedelta(days=months * 30)
            params["start_time"] = month_ago.strftime('%Y-%m-%d')
            params["end_time"] = now.strftime('%Y-%m-%d')
        # 今年
        elif "今年" in q:
            params["start_time"] = f"{now.year}-01-01"
            params["end_time"] = now.strftime('%Y-%m-%d')
        # 具体月份（仅在前面的规则都没匹配时才生效）
        m = re.search(r'(\d{1,2})月', q)
        if m and "start_time" not in params:
            month = int(m.group(1))
            params["start_time"] = f"{now.year}-{month:02d}-01"
            if month == 12:
                params["end_time"] = f"{now.year}-12-31"
            else:
                params["end_time"] = f"{now.year}-{month+1:02d}-01"

        # 提取部门参数（英文代码用词边界匹配，避免子串误匹配）
        dept_map = {
            'CI': 'CI', 'OM': 'OM', 'QA': 'QA', 'QC': 'QC',
            'FM': 'FM', 'DS': 'DS', 'FF': 'FF', 'EHS': 'EHS',
            'LG': 'LG', 'PD': 'PD', 'PM': 'PM', 'VM': 'VM',
        }
        for code in dept_map:
            if re.search(r'\b' + code + r'\b', q):
                params["dept"] = code
                break

        # 中文部门名映射
        dept_cn_map = {
            '运行保障部': 'OM', '设备管理部': 'FM', '质量保证部': 'QA',
            '原液生产部': 'DS', '自控信息部': 'CI', '质量控制部': 'QC',
            '验证管理部': 'VM', '制剂生产部': 'FF', '安全环保部': 'EHS',
            '物控部': 'LG', '采购部': 'PD', '生产管理办公室': 'PM',
        }
        for cn_name, code in dept_cn_map.items():
            if cn_name in q:
                params["dept"] = code
                break

        return params

    def _fix_time_range(self, sql: str, user_query: str, connection_name: str = None) -> str:
        """
        SQL后处理：检测用户查询中的时间意图，校验并修正SQL中的时间范围。
        解决LLM对"本周/本月"等时间关键词生成不一致的问题。
        """
        if not user_query or not sql:
            return sql

        from datetime import datetime, timedelta
        now = datetime.now()

        # 判断用户查询中的时间意图
        time_intent = None
        if "本周" in user_query:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            time_intent = {
                "start": this_week_start.strftime('%Y-%m-%d'),
                "end": now.strftime('%Y-%m-%d'),
                "label": "本周",
            }
        elif "上周" in user_query:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(days=1)
            time_intent = {
                "start": last_week_start.strftime('%Y-%m-%d'),
                "end": last_week_end.strftime('%Y-%m-%d'),
                "label": "上周",
            }
        elif "本月" in user_query:
            time_intent = {
                "start": now.replace(day=1).strftime('%Y-%m-%d'),
                "end": now.strftime('%Y-%m-%d'),
                "label": "本月",
            }
        elif "上月" in user_query:
            first_of_this_month = now.replace(day=1)
            last_month_end = first_of_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            time_intent = {
                "start": last_month_start.strftime('%Y-%m-%d'),
                "end": last_month_end.strftime('%Y-%m-%d'),
                "label": "上月",
            }
        elif "今年" in user_query:
            time_intent = {
                "start": f"{now.year}-01-01",
                "end": now.strftime('%Y-%m-%d'),
                "label": "今年",
            }

        if not time_intent:
            return sql

        expected_start = time_intent["start"]
        expected_end = time_intent["end"]
        label = time_intent["label"]

        # 检测SQL中的时间范围
        # 模式1: BETWEEN 'date1' AND 'date2'
        between_pattern = re.compile(
            r"(BETWEEN\s+)'(\d{4}-\d{2}-\d{2})'(\s+AND\s+)'(\d{4}-\d{2}-\d{2})'",
            re.IGNORECASE
        )
        # 模式2: >= 'date1' AND ... < 'date2' 或 > 'date1' AND ... <= 'date2'
        gte_lt_pattern = re.compile(
            r"(>=\s*|>\s*)'(\d{4}-\d{2}-\d{2})'(\s+AND\s+.*?)(<=\s*|<\s*)'(\d{4}-\d{2}-\d{2})'",
            re.IGNORECASE | re.DOTALL
        )

        fixed = False

        def replace_between(m):
            nonlocal fixed
            start_in_sql = m.group(2)
            end_in_sql = m.group(4)
            if start_in_sql != expected_start or end_in_sql != expected_end:
                # 检查是否偏差超过1天（允许今天日期的微小差异）
                try:
                    from datetime import datetime as dt
                    sql_start = dt.strptime(start_in_sql, '%Y-%m-%d')
                    expected_s = dt.strptime(expected_start, '%Y-%m-%d')
                    if abs((sql_start - expected_s).days) > 1:
                        logger.info(
                            f"[时间修正] {label}: SQL中{start_in_sql}→{expected_start}, "
                            f"{end_in_sql}→{expected_end}"
                        )
                        fixed = True
                        return f"{m.group(1)}'{expected_start}'{m.group(3)}'{expected_end}'"
                except ValueError:
                    pass
            return m.group(0)

        sql = between_pattern.sub(replace_between, sql)

        def replace_gte_lt(m):
            nonlocal fixed
            start_in_sql = m.group(2)
            end_in_sql = m.group(5)
            op1 = m.group(1).strip()
            op2 = m.group(4).strip()
            and_part = m.group(3)

            if start_in_sql != expected_start or end_in_sql != expected_end:
                try:
                    from datetime import datetime as dt
                    sql_start = dt.strptime(start_in_sql, '%Y-%m-%d')
                    expected_s = dt.strptime(expected_start, '%Y-%m-%d')
                    if abs((sql_start - expected_s).days) > 1:
                        logger.info(
                            f"[时间修正] {label}: SQL中{start_in_sql}→{expected_start}, "
                            f"{end_in_sql}→{expected_end}"
                        )
                        fixed = True
                        # 统一改为BETWEEN语法
                        return f"BETWEEN '{expected_start}' AND '{expected_end}'"
                except ValueError:
                    pass
            return m.group(0)

        sql = gte_lt_pattern.sub(replace_gte_lt, sql)

        # 检测DATEADD(WEEK,-1,...)或DATEADD(MONTH,-1,...)并替换
        dateadd_pattern = re.compile(
            r"DATEADD\s*\(\s*(WEEK|MONTH)\s*,\s*-\d+\s*,\s*GETDATE\s*\(\s*\)\s*\)",
            re.IGNORECASE
        )
        if dateadd_pattern.search(sql):
            logger.info(f"[时间修正] {label}: 替换DATEADD为BETWEEN '{expected_start}' AND '{expected_end}'")
            # 找到包含DATEADD的时间条件行，整体替换
            # 模式: xxx >= DATEADD(...) AND xxx < GETDATE()（可能有前导AND或没有）
            complex_pattern = re.compile(
                r"(?:AND\s+)?\w+(?:\.\w+)?\s*(?:>=|>)\s*DATEADD\s*\(\s*(?:WEEK|MONTH)\s*,\s*-\d+\s*,\s*GETDATE\s*\(\s*\)\s*\)\s+AND\s+\w+(?:\.\w+)?\s*(?:<=|<)\s*GETDATE\s*\(\s*\)",
                re.IGNORECASE
            )
            m = complex_pattern.search(sql)
            if m:
                matched = m.group(0)
                prefix = "AND " if matched.startswith("AND ") or matched.startswith("and ") else ""
                # 根据SQL是否查DW宽表选择时间字段
                is_dw = bool(re.search(r'(FROM|JOIN)\s+dw_', sql, re.IGNORECASE))
                time_field = "ApprovalTime" if is_dw else "wr.LASTSAVED"
                sql = complex_pattern.sub(
                    f"{prefix}{time_field} BETWEEN '{expected_start}' AND '{expected_end}'",
                    sql
                )
                fixed = True
            else:
                # 简单替换DATEADD本身
                sql = dateadd_pattern.sub(f"'{expected_start}'", sql)
                # 同时替换 GETDATE() 为结束日期（如果紧跟在<或<=后面）
                getdate_after_op = re.compile(
                    r"(<|<=)\s*GETDATE\s*\(\s*\)",
                    re.IGNORECASE
                )
                sql = getdate_after_op.sub(f"\\1 '{expected_end}'", sql)
                fixed = True

        # 如果用户有时间意图但SQL中没有任何时间条件，自动追加
        if not fixed and time_intent:
            # 检测SQL中是否有任何日期条件（BETWEEN/>=/>/DATEADD/GETDATE等）
            has_time_condition = bool(re.search(
                r"(BETWEEN|LASTSAVED\s*(?:>=|>|<=|<)|CREATEDTIME\s*(?:>=|>|<=|<)|"
                r"ApprovalTime\s*(?:>=|>|<=|<)|DATEADD|GETDATE\s*\(\s*\))",
                sql, re.IGNORECASE
            ))
            if not has_time_condition:
                # 判断SQL是否查询DW宽表（DW宽表用ApprovalTime，无别名前缀）
                is_dw_query = bool(re.search(
                    r'(FROM|JOIN)\s+dw_', sql, re.IGNORECASE
                )) or connection_name == "DW"
                if is_dw_query:
                    time_field = "ApprovalTime"
                else:
                    # 确定时间字段和表别名
                    alias = self._detect_wr_alias(sql)
                    time_field = f"{alias}.LASTSAVED"
                # 在WHERE子句末尾追加时间条件
                # 如果有WHERE，在最后一个条件后追加AND
                if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
                    # 在GROUP BY/ORDER BY/HAVING之前插入
                    insert_pos = len(sql)
                    for keyword in ['GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']:
                        kw_match = re.search(rf'\b{keyword}\b', sql, re.IGNORECASE)
                        if kw_match:
                            insert_pos = min(insert_pos, kw_match.start())
                            break
                    time_cond = f" AND {time_field} BETWEEN '{expected_start}' AND '{expected_end}' "
                    sql = sql[:insert_pos] + time_cond + sql[insert_pos:]
                else:
                    # 没有WHERE，在FROM...后添加WHERE
                    from_match = re.search(
                        r'(FROM\s+.*?(?:WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$))',
                        sql, re.IGNORECASE | re.DOTALL
                    )
                    if from_match:
                        insert_pos = from_match.end()
                        # 如果匹配到了WHERE等关键字，需要回退到它之前
                        for keyword in ['WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']:
                            kw_match = re.search(rf'\b{keyword}\b', sql[from_match.start():], re.IGNORECASE)
                            if kw_match:
                                insert_pos = from_match.start() + kw_match.start()
                                break
                        time_cond = f" WHERE {time_field} BETWEEN '{expected_start}' AND '{expected_end}' "
                        sql = sql[:insert_pos] + time_cond + sql[insert_pos:]
                    else:
                        # 兜底：直接追加
                        sql = sql.rstrip() + f" WHERE {time_field} BETWEEN '{expected_start}' AND '{expected_end}'"
                logger.info(f"[时间修正] {label}: SQL无时间条件，自动追加 BETWEEN '{expected_start}' AND '{expected_end}'")
                fixed = True

        return sql

    def _fix_count_logic(self, sql: str) -> str:
        """
        SQL后处理：修正工单计数逻辑。
        帆软报表中工单去重用 COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE))，
        LLM可能生成 COUNT(*) 或 COUNT(RECCODE) 导致重复计数。
        """
        if not sql:
            return sql

        # 检测是否涉及工单计数（FROM ATWORKFLOWRECORDS 或 DW宽表 dw_workorder_detail）
        has_work_order_table = bool(re.search(
            r'(FROM|JOIN)\s+ATWORKFLOWRECORDS', sql, re.IGNORECASE
        ))
        has_dw_workorder = bool(re.search(
            r'(FROM|JOIN)\s+dw_workorder_detail', sql, re.IGNORECASE
        ))
        if not has_work_order_table and not has_dw_workorder:
            return sql

        # DW宽表不需要别名前缀
        if has_dw_workorder and not has_work_order_table:
            dw_count = "COUNT(DISTINCT CONCAT(RECCODE, '|', RECFLOCODE))"
            count_star_pattern = re.compile(r'COUNT\s*\(\s*\*\s*\)', re.IGNORECASE)
            if count_star_pattern.search(sql):
                sql = count_star_pattern.sub(dw_count, sql)
                logger.info(f"[计数修正] DW宽表 COUNT(*) → {dw_count}")

            count_reccode_pattern = re.compile(
                r'COUNT\s*\(\s*(\w+\.)?RECCODE\s*\)', re.IGNORECASE
            )
            if count_reccode_pattern.search(sql):
                sql = count_reccode_pattern.sub(dw_count, sql)
                logger.info(f"[计数修正] DW宽表 COUNT(RECCODE) → {dw_count}")

            count_distinct_reccode_pattern = re.compile(
                r'COUNT\s*\(\s*DISTINCT\s+(\w+\.)?RECCODE\s*\)', re.IGNORECASE
            )
            if count_distinct_reccode_pattern.search(sql):
                sql = count_distinct_reccode_pattern.sub(dw_count, sql)
                logger.info(f"[计数修正] DW宽表 COUNT(DISTINCT RECCODE) → {dw_count}")

            return sql

        # 模式1: COUNT(*) → 替换为 COUNT(DISTINCT CONCAT(wr.RECCODE, '|', wr.RECFLOCODE))
        count_star_pattern = re.compile(
            r'COUNT\s*\(\s*\*\s*\)',
            re.IGNORECASE
        )
        if count_star_pattern.search(sql):
            # 确定表别名
            alias = self._detect_wr_alias(sql)
            new_count = f"COUNT(DISTINCT CONCAT({alias}.RECCODE, '|', {alias}.RECFLOCODE))"
            sql = count_star_pattern.sub(new_count, sql)
            logger.info(f"[计数修正] COUNT(*) → {new_count}")

        # 模式2: COUNT(wr.RECCODE) 或 COUNT(RECCODE) → 替换
        count_reccode_pattern = re.compile(
            r'COUNT\s*\(\s*(\w+\.)?RECCODE\s*\)',
            re.IGNORECASE
        )
        if count_reccode_pattern.search(sql):
            alias = self._detect_wr_alias(sql)
            new_count = f"COUNT(DISTINCT CONCAT({alias}.RECCODE, '|', {alias}.RECFLOCODE))"
            sql = count_reccode_pattern.sub(new_count, sql)
            logger.info(f"[计数修正] COUNT(RECCODE) → {new_count}")

        # 模式3: COUNT(DISTINCT wr.RECCODE) → 加上RECFLOCODE
        count_distinct_reccode_pattern = re.compile(
            r'COUNT\s*\(\s*DISTINCT\s+(\w+\.)?RECCODE\s*\)',
            re.IGNORECASE
        )
        if count_distinct_reccode_pattern.search(sql):
            alias = self._detect_wr_alias(sql)
            new_count = f"COUNT(DISTINCT CONCAT({alias}.RECCODE, '|', {alias}.RECFLOCODE))"
            sql = count_distinct_reccode_pattern.sub(new_count, sql)
            logger.info(f"[计数修正] COUNT(DISTINCT RECCODE) → {new_count}")

        return sql

    def _fix_group_by_for_employee(self, sql: str, user_query: str, connection_name: str = None) -> str:
        """
        SQL后处理：当用户查询"XX最多的员工"时，确保GROUP BY按员工分组。
        LLM可能错误地按工单类型(FLODESC)或其他维度分组。
        """
        if not sql or not user_query:
            return sql

        # 检测"XX最多的员工"意图
        employee_top_pattern = re.compile(
            r'(?:完成|处理|负责).*?(?:最多|最高|最大|第一).*?员工|'
            r'(?:最多|最高|最大|第一).*?(?:完成|处理|负责).*?员工|'
            r'(?:工单|数量).*?(?:最多|最高|最大).*?员工|'
            r'员工.*?(?:最多|最高|最大)|'
            r'谁.*?(?:最多|最高|最大|最忙)|'
            r'(?:哪个|哪位)员工',
            re.IGNORECASE
        )
        if not employee_top_pattern.search(user_query):
            return sql

        # 检查SQL是否有GROUP BY
        group_by_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if not group_by_match:
            return sql

        group_by_clause = group_by_match.group(1).strip()

        # 检查GROUP BY中是否包含员工相关字段（USRDESC/USRCODE/CREATEDBY等）
        has_employee_group = bool(re.search(
            r'USRDESC|USRCODE|CREATEDBY|员工|姓名|EMPLOYEE',
            group_by_clause, re.IGNORECASE
        ))

        if has_employee_group:
            # 已经按员工分组，检查是否有多余的分组字段
            # 如果GROUP BY同时包含FLODESC/RECFLOCODE等非员工字段，需要移除
            extra_fields = re.findall(
                r'(?:\w+\.)?(?:FLODESC|RECFLOCODE|RECSTATUS|RECFROMSTATUS|FLOWNAME|工单类型|流程名称|状态|审批节点|节点)',
                group_by_clause, re.IGNORECASE
            )
            if extra_fields:
                # 移除非员工分组字段
                cleaned = group_by_clause
                for field in extra_fields:
                    # 移除字段及其前导逗号
                    cleaned = re.sub(
                        r',?\s*(?:\w+\.)?' + re.escape(field) + r'\s*,?',
                        '', cleaned, flags=re.IGNORECASE
                    )
                    # 清理首尾逗号
                    cleaned = re.sub(r'^\s*,\s*', '', cleaned)
                    cleaned = re.sub(r'\s*,\s*$', '', cleaned)

                if cleaned != group_by_clause and cleaned.strip():
                    sql = sql.replace(group_by_clause, cleaned)
                    # 【修复Bug1】同步移除SELECT中对应的非员工非聚合字段
                    # 安全策略：只在最后一个SELECT（非CTE内部的SELECT）中移除
                    # 找到最后一个SELECT...FROM块（即CTE外的最终查询）
                    last_select_match = None
                    for m in re.finditer(
                        r'(SELECT\s+)(.*?)(\s+FROM\s+)',
                        sql, re.IGNORECASE | re.DOTALL
                    ):
                        last_select_match = m

                    if last_select_match:
                        select_text = last_select_match.group(2)
                        # 智能分割SELECT字段（考虑括号嵌套）
                        select_parts = self._split_select_fields(select_text)
                        new_parts = []
                        removed_names = []
                        for part in select_parts:
                            stripped = part.strip()
                            # 检查是否是聚合函数（保留）
                            if re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', stripped, re.IGNORECASE):
                                new_parts.append(stripped)
                                continue
                            # 检查是否是员工相关字段（保留）
                            if re.search(r'USRDESC|USRCODE|CREATEDBY|员工|姓名|EMPLOYEE', stripped, re.IGNORECASE):
                                new_parts.append(stripped)
                                continue
                            # 检查是否是需要移除的字段
                            should_remove = False
                            for field in extra_fields:
                                if re.search(r'(?:\w+\.)?' + re.escape(field), stripped, re.IGNORECASE):
                                    should_remove = True
                                    # 记录被移除的字段别名（如果有AS）
                                    as_match = re.search(r'\bAS\s+(\w+)', stripped, re.IGNORECASE)
                                    if as_match:
                                        removed_names.append(as_match.group(1))
                                    break
                            if not should_remove:
                                new_parts.append(stripped)

                        if len(new_parts) < len(select_parts):
                            new_select = ', '.join(new_parts)
                            sql = sql.replace(select_text, new_select)
                            # 同时移除ORDER BY中对被移除字段的引用
                            for rm_name in removed_names:
                                sql = re.sub(
                                    r',?\s*' + re.escape(rm_name) + r'\s*,?',
                                    '', sql, flags=re.IGNORECASE
                                )
                    logger.info(f"[分组修正] 移除非员工分组字段: {extra_fields} → GROUP BY {cleaned}")
            return sql

        # GROUP BY中没有员工字段，需要替换
        # 判断是否为DW宽表查询（DW宽表用EMPLOYEE_NAME，无别名前缀）
        is_dw_query = bool(re.search(r'(FROM|JOIN)\s+dw_', sql, re.IGNORECASE)) or connection_name == "DW"

        if is_dw_query:
            # DW宽表直接用EMPLOYEE_NAME字段
            new_group_by = "EMPLOYEE_NAME"
        else:
            # 查找SQL中是否有ATUSERS JOIN（有USRDESC可用）
            alias = self._detect_wr_alias(sql)
            has_atusers = bool(re.search(r'JOIN\s+ATUSERS', sql, re.IGNORECASE))

            if has_atusers:
                # 找到USRDESC的别名
                au_match = re.search(r'ATUSERS\s+(\w+)', sql, re.IGNORECASE)
                au_alias = au_match.group(1) if au_match else 'au'
                new_group_by = f"{au_alias}.USRDESC"
            else:
                # 没有JOIN ATUSERS，用CREATEDBY
                new_group_by = f"{alias}.CREATEDBY"

        # 替换GROUP BY
        sql = re.sub(
            r'GROUP\s+BY\s+.*?(?=\s+HAVING|\s+ORDER\s+BY|\s+LIMIT|$)',
            f'GROUP BY {new_group_by}',
            sql, flags=re.IGNORECASE | re.DOTALL
        )

        # 先替换SELECT中的非聚合字段为员工字段（必须在加TOP 1之前）
        # 找到SELECT和FROM之间的字段列表
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 智能分割：考虑括号嵌套，不在括号内的逗号才是字段分隔符
            parts = self._split_select_fields(select_clause)
            new_parts = []
            employee_field_added = False
            for part in parts:
                stripped = part.strip()
                # 保留聚合函数（COUNT/SUM/AVG/MAX/MIN）
                if re.search(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', stripped, re.IGNORECASE):
                    new_parts.append(stripped)
                else:
                    # 只添加一次员工字段（去重）
                    if not employee_field_added:
                        if is_dw_query:
                            new_parts.append("EMPLOYEE_NAME AS 员工姓名")
                        elif has_atusers:
                            new_parts.append(f"{au_alias}.USRDESC AS 员工姓名")
                        else:
                            new_parts.append(f"{alias}.CREATEDBY AS 员工姓名")
                        employee_field_added = True

            new_select = ', '.join(new_parts)
            sql = sql.replace(select_clause, new_select)
            logger.info(f"[分组修正] SELECT和GROUP BY替换为员工字段")

        # 确保ORDER BY按工单数量降序
        if not re.search(r'ORDER\s+BY', sql, re.IGNORECASE):
            # 找到聚合字段的别名（如"工单数量"）
            count_alias_match = re.search(
                r'COUNT\s*\([^)]+\)\s+AS\s+(\w+)',
                sql, re.IGNORECASE
            )
            order_field = count_alias_match.group(1) if count_alias_match else '工单数量'
            sql = sql.rstrip() + f' ORDER BY {order_field} DESC'
        elif re.search(r'ORDER\s+BY', sql, re.IGNORECASE) and not re.search(r'DESC', sql, re.IGNORECASE):
            # 有ORDER BY但没有DESC，追加DESC
            sql = re.sub(
                r'(ORDER\s+BY\s+.*?)(?:\s+LIMIT|\s*$)',
                r'\1 DESC',
                sql, flags=re.IGNORECASE
            )

        # 如果用户问"最多"且SQL没有TOP，加上TOP 1（必须在SELECT字段替换之后）
        if re.search(r'最多|第一|最高', user_query, re.IGNORECASE):
            if not re.search(r'\bTOP\s+\d+', sql, re.IGNORECASE):
                sql = re.sub(r'\bSELECT\b', 'SELECT TOP 1', sql, count=1, flags=re.IGNORECASE)

        return sql

    @staticmethod
    def _split_select_fields(select_clause: str) -> List[str]:
        """智能分割SELECT字段列表，考虑括号嵌套和引号内的逗号"""
        parts = []
        current = []
        paren_depth = 0
        in_quote = False
        quote_char = None
        for ch in select_clause:
            if in_quote:
                current.append(ch)
                if ch == quote_char:
                    in_quote = False
                continue
            if ch in ("'", '"'):
                in_quote = True
                quote_char = ch
                current.append(ch)
                continue
            if ch == '(':
                paren_depth += 1
                current.append(ch)
            elif ch == ')':
                paren_depth -= 1
                current.append(ch)
            elif ch == ',' and paren_depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _detect_wr_alias(self, sql: str) -> str:
        """检测ATWORKFLOWRECORDS表的别名"""
        # 模式: FROM ATWORKFLOWRECORDS wr
        m = re.search(
            r'(?:FROM|JOIN)\s+ATWORKFLOWRECORDS\s+(\w+)',
            sql, re.IGNORECASE
        )
        if m:
            return m.group(1)
        # 无别名时检查SQL中使用的别名前缀
        m = re.search(r'(\w+)\.RECCODE', sql, re.IGNORECASE)
        if m:
            return m.group(1)
        return 'wr'

    def query(self, user_query: str, context: dict = None) -> Generator:
        # 保存当前查询，供_fix_time_range使用
        self._current_user_query = user_query

        if not self.llm_service or not self.llm_service.client:
            yield {"type": "done", "summary": "LLM服务未初始化，请检查API Key配置"}
            return

        yield {"type": "thinking", "content": "正在分析您的问题..."}

        # ===== 优化1: 语义缓存查找 =====
        try:
            cache = get_semantic_cache()
            cached = cache.lookup(user_query)
            if cached and cached.get("sql"):
                logger.info(f"[AgentQueryEngine] 语义缓存命中，直接执行缓存SQL")
                yield {"type": "thinking", "content": "命中缓存，直接执行查询..."}
                cached_sql = cached["sql"]
                # 优先使用缓存中存储的连接名，降级到SQL推断
                conn = cached.get("connection_name", "")
                if not conn:
                    conn = "DW"
                    if "ATWORKFLOWRECORDS" in cached_sql or "ATUSERS" in cached_sql or "ATWORKFLOWSTATUS" in cached_sql:
                        conn = "EAM"
                    elif "WMS" in cached_sql.upper() or "ZXJT_WMSXCL" in cached_sql or "INV_LOT_LOC_ID" in cached_sql:
                        conn = "WMS_PROD"
                result = self._execute_sql(cached_sql, conn)
                if result.get("success"):
                    data = result["data"]
                    yield {"type": "sql", "sql": cached_sql, "step": 1, "source": "cache"}
                    yield {
                        "type": "data", "data": data, "step": 1,
                        "total_rows": result.get("total_rows", len(data)),
                        "elapsed": result.get("elapsed", 0),
                    }
                    if data:
                        auto_chart = self._auto_generate_chart(data, user_query)
                        if auto_chart:
                            yield auto_chart
                        # 工单时序数据深度分析
                        ts_event = self._try_timeseries_analysis(data, user_query)
                        if ts_event:
                            yield ts_event
                    # 直接生成总结
                    yield {"type": "thinking", "content": "正在生成回答..."}
                    yield from self._generate_summary(user_query, data)
                    return
                else:
                    logger.info("[AgentQueryEngine] 缓存SQL执行失败，降级到LLM推理")
        except Exception as e:
            logger.warning(f"[AgentQueryEngine] 语义缓存查找异常: {e}")

        # ===== 优化2: 本地模板路由（跳过LLM推理轮） =====
        local_route = self._try_local_route(user_query)
        if local_route:
            template_name = local_route["template_name"]
            logger.info(f"[AgentQueryEngine] 本地路由命中模板: {template_name}")

            yield {"type": "thinking", "content": f"匹配到模板: {template_name}，直接执行..."}

            template_info = self._template_store.get(template_name)
            if template_info:
                raw_sql = template_info['sql']
                connection_name = template_info['connection']

                # 提取参数并填充模板
                template_params = self._extract_time_params(user_query)
                filled_sql = self._fill_template_params(raw_sql, template_params)

                # ⚠️ 本地路由SQL后处理：员工分组修正 + 计数修正
                filled_sql = self._fix_group_by_for_employee(filled_sql, user_query)
                filled_sql = self._fix_count_logic(filled_sql)

                yield {"type": "sql", "sql": filled_sql, "step": 1, "template": template_name}

                # 执行SQL
                result = self._execute_sql(filled_sql, connection_name)

                if result.get("success"):
                    data = result["data"]
                    yield {
                        "type": "data", "data": data, "step": 1,
                        "total_rows": result.get("total_rows", len(data)),
                        "elapsed": result.get("elapsed", 0),
                        "template": template_name,
                    }
                    if data:
                        auto_chart = self._auto_generate_chart(data, user_query)
                        if auto_chart:
                            yield auto_chart
                        # 工单时序数据深度分析
                        ts_event = self._try_timeseries_analysis(data, user_query)
                        if ts_event:
                            yield ts_event

                    # 存入语义缓存
                    try:
                        cache = get_semantic_cache()
                        cache.store(user_query, {"template": template_name, "params": template_params},
                                   filled_sql, verified=True, connection_name=connection_name)
                    except Exception:
                        pass

                    # 生成总结
                    yield {"type": "thinking", "content": "正在生成回答..."}
                    yield from self._generate_summary(user_query, data if result.get("success") else [])
                    return
                else:
                    error_msg = result.get("error", "未知错误")
                    logger.warning(f"[AgentQueryEngine] 模板SQL执行失败: {error_msg}，降级到LLM推理")

        # ===== 原有LLM推理流程 =====

        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        if context:
            extra_info = context.get("extra_context", "")
            if extra_info:
                messages[-1]["content"] += f"\n\n补充上下文: {extra_info}"

        steps: List[AgentStep] = []
        consecutive_llm_errors = 0

        for round_num in range(MAX_AGENT_ROUNDS):
            logger.info(f"[AgentQueryEngine] 第{round_num + 1}轮推理")

            try:
                response = self.llm_service.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=2000
                )
            except Exception as e:
                logger.error(f"[AgentQueryEngine] LLM调用失败: {e}")
                consecutive_llm_errors += 1
                if consecutive_llm_errors >= 2:
                    yield {"type": "text", "content": f"AI服务连续{consecutive_llm_errors}次调用失败，请稍后重试或检查API账户状态。"}
                    yield {"type": "done", "summary": "查询失败"}
                    return
                yield {"type": "text", "content": f"AI服务调用失败: {e}"}
                yield {"type": "done", "summary": "查询失败"}
                return

            if "error" in response and response.get("content") is None:
                consecutive_llm_errors += 1
                if consecutive_llm_errors >= 2:
                    yield {"type": "text", "content": f"AI服务连续异常({consecutive_llm_errors}次): {response['error']}。请检查API账户是否欠费。"}
                    yield {"type": "done", "summary": "查询失败"}
                    return
                yield {"type": "text", "content": f"AI服务异常: {response['error']}"}
                yield {"type": "done", "summary": "查询失败"}
                return

            consecutive_llm_errors = 0

            tool_calls = response.get("tool_calls")

            if tool_calls:
                assistant_msg = {"role": "assistant", "content": response.get("content")}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function_name"],
                                "arguments": tc.get("raw_arguments", safe_json_dumps(tc["arguments"], ensure_ascii=False))
                            }
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                for tc in tool_calls:
                    tool_call_id = tc["id"]
                    tool_name = tc["function_name"]
                    tool_args = tc["arguments"]

                    step = AgentStep(
                        step=len(steps) + 1,
                        tool_name=tool_name,
                        tool_input=tool_args
                    )
                    steps.append(step)

                    logger.info(
                        f"[AgentQueryEngine] 调用工具: {tool_name} | "
                        f"步骤={step.step} | 参数={safe_json_dumps(tool_args, ensure_ascii=False)[:200]}"
                    )

                    tool_result_data = None
                    tool_result_scada = None
                    for event in self._handle_tool_call(tool_name, tool_args, step.step):
                        yield event
                        if isinstance(event, dict):
                            if event["type"] == "data":
                                tool_result_data = event
                            elif event["type"] == "scada_analysis":
                                ar = event.get("analysis_result", {})
                                analysis_type = event.get("analysis_type", "raw")
                                if analysis_type == "threshold":
                                    summary_text = (
                                        f"SCADA阈值分析结果: 超标{ar.get('exceed_count', 0)}次，"
                                        f"累计持续{ar.get('total_duration_min', 0)}分钟，"
                                        f"最长持续{ar.get('max_duration_min', 0)}分钟。"
                                        f"原始结果: {safe_json_dumps({k: v for k, v in ar.items() if k != 'periods'}, ensure_ascii=False)}"
                                    )
                                elif analysis_type == "comparison":
                                    devices = ar.get("devices", {})
                                    summary_text = f"SCADA对比分析结果: 共{len(devices)}个设备对比。"
                                    for tn, st in devices.items():
                                        summary_text += (
                                            f"{st.get('label', tn)}: 当前{st.get('current', 'N/A')}{st.get('unit', '')}, "
                                            f"平均{st.get('avg', 'N/A')}{st.get('unit', '')}。"
                                        )
                                    summary_text += f"原始结果: {safe_json_dumps({k: v for k, v in ar.items() if k not in ('devices', 'chart_series')}, ensure_ascii=False)}"
                                elif analysis_type == "trend":
                                    summary_text = (
                                        f"SCADA趋势分析结果: 趋势{ar.get('trend', '未知')}, "
                                        f"总变化{ar.get('total_change', 0)}, "
                                        f"每小时变化{ar.get('rate_per_hour', 0)}。"
                                        f"原始结果: {safe_json_dumps({k: v for k, v in ar.items() if k != 'hourly_avg'}, ensure_ascii=False)}"
                                    )
                                else:
                                    summary_text = event.get("summary", "SCADA原始数据查询完成")
                                tool_result_scada = {"type": "scada_result", "summary": summary_text}

                    if tool_result_data is not None:
                        result_content = safe_json_dumps(tool_result_data, ensure_ascii=False)
                        if tool_result_data.get("data") is not None and len(tool_result_data.get("data", [])) == 0:
                            result_content += "\n\n⚠️ 查询返回0条数据。可能原因：1)WHERE条件中使用了不存在的字段 2)筛选条件过于严格 3)时间范围内没有数据。请修改SQL重试，建议：放宽筛选条件、去掉不存在的字段、扩大时间范围。"

                        if tool_name == "execute_sql" and tool_result_data.get("data") and len(tool_result_data.get("data", [])) > 0:
                            auto_chart = self._auto_generate_chart(tool_result_data["data"], user_query)
                            if auto_chart:
                                yield auto_chart
                            # 工单时序数据深度分析
                            ts_event = self._try_timeseries_analysis(tool_result_data["data"], user_query)
                            if ts_event:
                                yield ts_event
                    elif tool_result_scada is not None:
                        result_content = safe_json_dumps(tool_result_scada, ensure_ascii=False)
                    else:
                        result_content = "工具执行完成但无返回数据"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content[:4000]
                    })
            else:
                final_content = response.get("content", "")

                if final_content:
                    yield {"type": "thinking", "content": "正在生成回答..."}

                    data_facts = []
                    has_valid_data = False
                    for msg in messages:
                        if msg.get("role") == "tool" and msg.get("content"):
                            content = msg["content"]
                            try:
                                tool_data = json.loads(content)
                                if isinstance(tool_data, list):
                                    # execute_sql返回的是JSON数组
                                    if tool_data:
                                        rows = tool_data[:20]
                                        data_facts.append(safe_json_dumps(rows, ensure_ascii=False))
                                        has_valid_data = True
                                    else:
                                        data_facts.append("[该步骤查询返回0条数据，请忽略此步骤]")
                                elif isinstance(tool_data, dict):
                                    # scada_analysis工具结果
                                    if tool_data.get("type") == "scada_result":
                                        data_facts.append(tool_data.get("summary", ""))
                                        has_valid_data = True
                                    elif tool_data.get("data") is not None:
                                        rows = tool_data["data"][:20] if isinstance(tool_data["data"], list) else [tool_data["data"]]
                                        if rows:
                                            data_facts.append(safe_json_dumps(rows, ensure_ascii=False))
                                            has_valid_data = True
                                        else:
                                            data_facts.append("[该步骤查询返回0条数据，请忽略此步骤]")
                                    if tool_data.get("error"):
                                        data_facts.append(f"[查询错误: {tool_data['error'][:200]}]")
                            except (json.JSONDecodeError, TypeError):
                                # 非JSON内容（如SCADA分析摘要文本）
                                if content and not content.startswith("SQL执行失败"):
                                    data_facts.append(content[:2000])
                                    # 如果内容包含具体数字，认为有有效数据
                                    if re.search(r'\d+\.?\d*', content):
                                        has_valid_data = True

                    if not data_facts:
                        full_text = "抱歉，未能获取到任何数据来回答您的问题。"
                        yield {"type": "text", "content": full_text}
                        yield {"type": "done", "summary": full_text[:200]}
                        return

                    summary_prompt = (
                        f"用户问题: {user_query}\n\n"
                        f"以下是查询获取到的数据结果:\n"
                    )
                    for i, fact in enumerate(data_facts):
                        summary_prompt += f"\n--- 数据{i+1} ---\n{fact}\n"
                    if has_valid_data:
                        summary_prompt += (
                            "\n请基于以上数据结果，用简洁的中文直接回答用户的问题。"
                            "必须使用数据中的具体数字，不要用XX代替。"
                            "如果某些步骤返回0条数据，请忽略该步骤，只基于有数据的步骤回答。"
                        )
                    else:
                        summary_prompt += (
                            "\n所有查询都返回了0条数据。请如实说明没有找到符合条件的数据，"
                            "并建议用户调整查询条件（如扩大时间范围、去掉部门筛选等）。"
                        )

                    stream_messages = [
                        {"role": "system", "content": "你是数据分析助手，请基于查询结果数据回答问题，必须使用具体数字。"},
                        {"role": "user", "content": summary_prompt},
                    ]

                    full_text = ""
                    for chunk in self.llm_service.chat_stream(
                        stream_messages,
                        max_tokens=1000,
                        temperature=0.1
                    ):
                        if chunk and not chunk.startswith("[错误]"):
                            full_text += chunk
                            yield {"type": "text", "content": chunk}

                    yield {"type": "done", "summary": full_text[:200]}
                else:
                    yield {"type": "done", "summary": "未能获取有效回答"}

                return

        yield {"type": "thinking", "content": "已达到最大推理轮数，正在总结已有结果..."}

        data_facts = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("content"):
                content = msg["content"]
                try:
                    tool_data = json.loads(content)
                    if isinstance(tool_data, list):
                        if tool_data:
                            data_facts.append(safe_json_dumps(tool_data[:20], ensure_ascii=False))
                    elif isinstance(tool_data, dict):
                        if tool_data.get("type") == "scada_result":
                            data_facts.append(tool_data.get("summary", ""))
                        elif tool_data.get("data"):
                            rows = tool_data["data"][:20] if isinstance(tool_data["data"], list) else [tool_data["data"]]
                            data_facts.append(safe_json_dumps(rows, ensure_ascii=False))
                        elif tool_data.get("error"):
                            data_facts.append(f"[查询错误: {tool_data['error']}]")
                except (json.JSONDecodeError, TypeError):
                    if content and not content.startswith("SQL执行失败"):
                        data_facts.append(content[:2000])

        summary_prompt = f"用户问题: {user_query}\n\n以下是查询获取到的数据结果:\n"
        for i, fact in enumerate(data_facts):
            summary_prompt += f"\n--- 数据{i+1} ---\n{fact}\n"
        summary_prompt += "\n请基于以上数据结果，用简洁的中文总结回答用户的问题。必须使用数据中的具体数字。"

        summary_messages = [
            {"role": "system", "content": "你是数据分析助手，请基于查询结果数据回答问题，必须使用具体数字。"},
            {"role": "user", "content": summary_prompt},
        ]

        full_text = ""
        for chunk in self.llm_service.chat_stream(summary_messages, max_tokens=800):
            if chunk and not chunk.startswith("[错误]"):
                full_text += chunk
                yield {"type": "text", "content": chunk}

        yield {"type": "done", "summary": full_text[:200]}


_engine_instance: Optional[AgentQueryEngine] = None


def get_agent_engine() -> AgentQueryEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AgentQueryEngine()
    return _engine_instance
