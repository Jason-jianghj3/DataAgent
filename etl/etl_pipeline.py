"""
DataAgentDW ETL 管道

从业务源库(EAM/WMS_PROD/EKP/HISTORIAN)抽取数据，
经清洗转换后写入 DataAgentDW 宽表，供 LLM 生成简单 SELECT 查询。
"""
import os
import time
from datetime import datetime
from decimal import Decimal

import pymssql
import pymysql

from utils.logger import logger
from utils.db_config import get_db_config
from utils.serialization import convert_row_types


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BATCH_SIZE = 500
DW_DB_NAME = os.getenv("DB_DW_DATABASE", "DataAgentDW")

# ETL 运行状态（模块级单例）
_etl_status = {
    "last_run": None,
    "last_success": None,
    "tasks": {},
}


# ---------------------------------------------------------------------------
# 源查询 SQL（已清洗，去除帆软 ${} 模板语法）
# ---------------------------------------------------------------------------

SQL_WORKORDER_DETAIL = """
WITH Ordered AS (
    SELECT RECCODE, RECFLOCODE, RECFROMSTATUS, CREATEDBY, RECFLONODE AS OldNode, LASTSAVED,
        ROW_NUMBER() OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY LASTSAVED, RECCODE) AS NewNode
    FROM ATWORKFLOWRECORDS
),
ProcessTime AS (
    SELECT RECCODE, RECFLOCODE, RECFROMSTATUS, CREATEDBY, NewNode, LASTSAVED AS ApprovalTime,
        LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode) AS CreateTime,
        CASE WHEN LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode) IS NOT NULL
            THEN DATEDIFF(HOUR, LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode), LASTSAVED)
            ELSE NULL END AS ProcessHours,
        CASE WHEN LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode) IS NOT NULL
            THEN ROUND(DATEDIFF(MINUTE, LAG(LASTSAVED) OVER (PARTITION BY RECCODE, RECFLOCODE ORDER BY NewNode), LASTSAVED) / 1440.0, 2)
            ELSE NULL END AS ProcessDays
    FROM Ordered
),
data_all AS (
    SELECT RECCODE, RECFLOCODE, RECFROMSTATUS, CREATEDBY, CreateTime, ApprovalTime, ProcessHours, ProcessDays
    FROM ProcessTime WHERE CreateTime IS NOT NULL AND ProcessDays > 0
)
SELECT da.RECCODE, da.RECFLOCODE, af.FLODESC, af.FLOENTITYDESC,
    da.CREATEDBY, au.USRDESC AS EMPLOYEE_NAME, au.USRMRC AS DEPT_CODE, au.USRMRCDESC AS DEPT_NAME,
    da.CreateTime, da.ApprovalTime, da.ProcessHours, da.ProcessDays,
    CASE
        WHEN da.ProcessHours <= 3 THEN N'3小时内'
        WHEN da.ProcessDays <= 1 THEN N'3小时到1天'
        WHEN da.ProcessDays <= 5 THEN N'1天到5天'
        WHEN da.ProcessDays <= 10 THEN N'5天到10天'
        WHEN da.ProcessDays <= 30 THEN N'10天到30天'
        WHEN da.ProcessDays <= 90 THEN N'30天到90天'
        WHEN da.ProcessDays > 90 THEN N'大于90天'
        ELSE N'未分类'
    END AS ProcessTimeCategory,
    da.RECFROMSTATUS
FROM data_all da
LEFT JOIN ATUSERS au ON da.CREATEDBY = au.USRCODE
LEFT JOIN ATWORKFLOW af ON da.RECFLOCODE = af.FLOCODE
WHERE au.USRDESC IS NOT NULL AND au.USRMRC IS NOT NULL AND au.USRMRC <> '*'
"""

SQL_WORKORDER_COMPLETION = """
WITH workflow_status AS (
    SELECT af.wfscode, af.WFSFLOCODE, af.WFSSTATUS, at.FLODESC AS flow_type, af.LASTSAVED,
        CASE WHEN RIGHT(af.WFSSTATUS, 2) = 'WC' THEN N'已完成' ELSE N'未完成' END AS completion_status
    FROM ATWORKFLOWSTATUS af
    LEFT JOIN ATWORKFLOWNODES aw ON af.WFSFLOCODE = aw.NODCODE AND af.WFSSTATUS = aw.NODEAMSTATUS
    LEFT JOIN ATWORKFLOW at ON at.FLOCODE = af.WFSFLOCODE AND at.FLOCODE = aw.NODCODE
    WHERE aw.noddesc NOT LIKE N'%取消%' AND af.WFSSTATUS IS NOT NULL AND at.FLODESC IS NOT NULL
)
SELECT wfscode, WFSFLOCODE, LEFT(flow_type, 7) AS flow_type, completion_status, LASTSAVED
FROM workflow_status
"""

SQL_WORKORDER_PENDING = """
SELECT r.CREATEDBY, ats.USRMRC AS DEPT_CODE, r.WFSSTATUS, AW.noddesc, r.WFSENTITY, r.wfscode,
    a.FLODESC AS WFSFLOCODE, r.LASTSAVED, r.CurrentApproverID, r.CurrentApprover, r.CurrentDepartment,
    CAST(DATEDIFF(MINUTE, r.LASTSAVED, GETDATE()) AS DECIMAL(10, 4)) / 1440.0 AS DaysSinceLastSave
FROM ATWORKFLOWSTATUS r
LEFT JOIN ATWORKFLOW a ON r.WFSFLOCODE = a.FLOCODE
LEFT JOIN ATUSERS ats ON ats.USRCODE = r.CREATEDBY
LEFT JOIN ATWORKFLOWNODES aw ON R.WFSFLOCODE = aw.NODCODE AND R.WFSSTATUS = aw.NODEAMSTATUS
WHERE CurrentApproverID IS NOT NULL AND CurrentApproverID <> ''
"""

SQL_DEPARTMENT = """
SELECT DISTINCT AU.USRMRC as dept_code, AU.USRMRCDESC as dept_name
FROM ATUSERS AU WHERE AU.USRMRC IS NOT NULL AND AU.USRMRC <> '*'
ORDER BY AU.USRMRC
"""

SQL_FLOW_CATEGORY = """
SELECT DISTINCT FLOENTITYDESC as flow_category, au.USRMRC as dept_code
FROM ATWORKFLOW AT LEFT JOIN ATWORKFLOWRECORDS af ON af.RECFLOCODE = AT.FLOCODE
LEFT JOIN ATUSERS au ON af.CREATEDBY = au.USRCODE
WHERE AT.FLOCODE IS NOT NULL and au.USRMRC IS NOT NULL and au.USRMRC <> '*'
ORDER BY FLOENTITYDESC
"""

SQL_FLOW_TYPE = """
SELECT DISTINCT FLODESC as flow_type, FLOENTITYDESC, AU.USRMRC as dept_code
FROM ATWORKFLOW AT LEFT JOIN ATWORKFLOWRECORDS af ON af.RECFLOCODE = AT.FLOCODE
LEFT JOIN ATUSERS au ON af.CREATEDBY = au.USRCODE
WHERE FLODESC IS NOT NULL and au.USRMRC is not null and au.USRMRC <> '*'
ORDER BY FLODESC
"""

SQL_INVENTORY_ERP = """
SELECT 'ERP' AS source, '01' AS zone_id, N'研发区' AS zone_name,
    COALESCE(SUM(CASE WHEN WAREHOUSEID = '01' THEN qty ELSE 0 END), 0) AS qty FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '02', N'物资物料区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '02' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '03', N'原液产品区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '03' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '04', N'成品区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '04' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '05', N'不合格品区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '05' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '08', N'DS库区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '08' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '09', N'FF库区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '09' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '10', N'EM库区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '10' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
UNION ALL
SELECT 'ERP', '11', N'QC库区', COALESCE(SUM(CASE WHEN WAREHOUSEID = '11' THEN qty ELSE 0 END), 0) FROM zxncc.wmsxcl
"""

SQL_INVENTORY_WMS = """
SELECT 'WMS' AS source, Z.zoneid AS zone_id,
    CASE Z.zoneid WHEN '01' THEN N'研发区' WHEN '02' THEN N'物资物料区' WHEN '03' THEN N'原液产品区'
        WHEN '04' THEN N'成品区' WHEN '05' THEN N'不合格品区' WHEN '08' THEN N'DS库区'
        WHEN '09' THEN N'FF库区' WHEN '10' THEN N'EM库区' WHEN '11' THEN N'QC库区'
        ELSE N'其他' END AS zone_name,
    SUM(Z.fmqty) AS qty
FROM ZXJT_WMSXCL Z
WHERE Z.zoneid IN ('01','02','03','04','05','08','09','10','11')
GROUP BY Z.zoneid
"""

SQL_MATERIAL_STAGNANT = """
SELECT * FROM (
    SELECT CASE WHEN bs.sku_group5 IN (N'C类物资', N'B类物资') THEN N'B、C类物资'
                WHEN bs.sku_group5 IN (N'包材', N'原料', N'辅料') THEN N'物料'
                ELSE bs.sku_group5 END AS material_type,
        bs.skudescr1 AS sku_descr, SUM(illi.qty) as sum_qty, ila.lotAtt06,
        bs.alternate_sku3, t.uomDescr,
        illi.warehouseId, bl.zoneDescr AS warehouse_name,
        ISNULL(DATEDIFF(DAY, last_out.last_out_time, GETDATE()), 180) AS days_no_out
    FROM dbo.INV_LOT_LOC_ID illi WITH (NOLOCK)
    LEFT JOIN dbo.INV_LOT_ATT ila ON ila.organizationId = illi.organizationId AND ila.lotNum = illi.lotNum
    LEFT JOIN dbo.BAS_SKU bs WITH (NOLOCK) ON bs.organizationId = illi.organizationId AND bs.customerId = illi.customerId AND bs.sku = illi.sku
    LEFT JOIN dbo.BAS_LOCATION bl WITH (NOLOCK) ON bl.organizationId = illi.organizationId AND bl.warehouseId = illi.warehouseId AND bl.locationId = illi.locationId
    LEFT JOIN (SELECT atl.organizationId, atl.warehouseId, ila1.lotAtt06
               FROM dbo.ACT_TRANSACTION_LOG atl WITH (NOLOCK)
               LEFT JOIN dbo.INV_LOT_ATT ila1 WITH (NOLOCK) ON ila1.organizationId = atl.organizationId AND ila1.lotNum = atl.toLotNum
               WHERE atl.status = '99' AND atl.transactionType IN ('IN', 'SO') AND atl.transactionTime > DATEADD(DAY, -180, GETDATE())) atl
        ON atl.organizationId = illi.organizationId AND atl.warehouseId = illi.warehouseId AND atl.lotAtt06 = ila.lotAtt06
    LEFT JOIN (SELECT atl2.organizationId, atl2.warehouseId, ila2.lotAtt06, MAX(atl2.transactionTime) AS last_out_time
               FROM dbo.ACT_TRANSACTION_LOG atl2 WITH (NOLOCK)
               LEFT JOIN dbo.INV_LOT_ATT ila2 WITH (NOLOCK) ON ila2.organizationId = atl2.organizationId AND ila2.lotNum = atl2.toLotNum
               WHERE atl2.status = '99' AND atl2.transactionType IN ('IN', 'SO')
               GROUP BY atl2.organizationId, atl2.warehouseId, ila2.lotAtt06) last_out
        ON last_out.organizationId = illi.organizationId AND last_out.warehouseId = illi.warehouseId AND last_out.lotAtt06 = ila.lotAtt06
    LEFT JOIN BAS_SKU_MULTIWAREHOUSE bsm WITH(NOLOCK) ON ila.organizationId = bsm.organizationId AND illi.warehouseId = bsm.warehouseId AND ila.customerId = bsm.customerId AND bs.sku = bsm.sku
    LEFT JOIN BAS_PACKAGE_DETAILS t WITH(NOLOCK) ON ila.organizationId = t.organizationId
        AND CASE WHEN isnull(bsm.customerId, '') = '' THEN bs.customerId ELSE bsm.customerId END = t.customerId
        AND CASE WHEN isnull(bsm.packId, '') = '' THEN bs.packId ELSE bsm.packId END = t.packId
        AND CASE WHEN isnull(bsm.reportUom, '') = '' THEN bs.reportUom ELSE bsm.reportUom END = t.uom
    WHERE illi.organizationId = 'ZXJT' AND bl.zoneid NOT IN ('06','07')
        AND illi.qty > 0 AND atl.lotAtt06 IS NULL
    GROUP BY bs.sku_group5, bs.skudescr1, ila.lotAtt06, bs.alternate_sku3, t.uomDescr, illi.warehouseId, bl.zoneDescr, last_out.last_out_time
) t
"""

SQL_MATERIAL_EXPIRY = """
SELECT * FROM (
    SELECT CASE WHEN bs.sku_group5 IN (N'C类物资', N'B类物资') THEN N'B、C类物资'
                WHEN bs.sku_group5 IN (N'包材', N'原料', N'辅料') THEN N'物料'
                ELSE bs.sku_group5 END AS material_type,
        CONVERT(NVARCHAR(100), ila.lotAtt04, 23) AS expiry_date,
        ila.lotAtt06,
        bs.alternate_sku3,
        bs.skudescr1 AS sku_descr,
        t.uomDescr,
        SUM(illi.qty) AS sum_qty,
        DATEDIFF(DAY, GETDATE(), ila.lotAtt04) AS days_to_expiry
    FROM dbo.INV_LOT_LOC_ID illi WITH (NOLOCK)
    LEFT JOIN dbo.INV_LOT_ATT ila ON ila.organizationId = illi.organizationId AND ila.lotNum = illi.lotNum
    LEFT JOIN dbo.BAS_SKU bs WITH (NOLOCK) ON bs.organizationId = illi.organizationId AND bs.customerId = illi.customerId AND bs.sku = illi.sku
    LEFT JOIN dbo.BAS_LOCATION bl WITH (NOLOCK) ON bl.organizationId = illi.organizationId AND bl.warehouseId = illi.warehouseId AND bl.locationId = illi.locationId
    LEFT JOIN BAS_SKU_MULTIWAREHOUSE bsm WITH(NOLOCK) ON ila.organizationId = bsm.organizationId AND illi.warehouseId = bsm.warehouseId AND ila.customerId = bsm.customerId AND bs.sku = bsm.sku
    LEFT JOIN BAS_PACKAGE_DETAILS t WITH(NOLOCK) ON ila.organizationId = t.organizationId
        AND CASE WHEN isnull(bsm.customerId, '') = '' THEN bs.customerId ELSE bsm.customerId END = t.customerId
        AND CASE WHEN isnull(bsm.packId, '') = '' THEN bs.packId ELSE bsm.packId END = t.packId
        AND CASE WHEN isnull(bsm.reportUom, '') = '' THEN bs.reportUom ELSE bsm.reportUom END = t.uom
    WHERE illi.organizationId = 'ZXJT'
        AND bl.zoneid NOT IN ('06','07')
        AND illi.qty > 0
        AND ila.lotAtt04 IS NOT NULL
        AND DATEDIFF(DAY, GETDATE(), ila.lotAtt04) BETWEEN 0 AND 180
    GROUP BY bs.sku_group5, ila.lotAtt04, ila.lotAtt06, bs.alternate_sku3, bs.skudescr1, t.uomDescr
) t
ORDER BY days_to_expiry
"""

SQL_INSPECTION_ANOMALY = """
SELECT CONVERT(NVARCHAR(20), ad.orderDate, 23) AS order_date,
    ad.asnNo AS asn_no, ad.docType AS doc_type,
    iz.zoneName AS zone_name,
    bs.skudescr1 AS material_name, bs.alternate_sku3 AS material_code,
    ad.toLotNum AS production_batch, ila.lotAtt06 AS custom_batch,
    CASE WHEN ad.docResult = '1' THEN N'合格' WHEN ad.docResult = '2' THEN N'不合格' ELSE N'待检' END AS quality_status,
    ad.qty, ad.receiverId AS warehouse_keeper,
    ad.receiverId AS receiver, ad.reviewerId AS reviewer,
    ad.remark, 1 AS is_anomaly
FROM dbo.ACT_TRANSACTION_LOG ad WITH (NOLOCK)
LEFT JOIN dbo.BAS_SKU bs WITH (NOLOCK) ON bs.organizationId = ad.organizationId AND bs.customerId = ad.customerId AND bs.sku = ad.sku
LEFT JOIN dbo.INV_LOT_ATT ila WITH (NOLOCK) ON ila.organizationId = ad.organizationId AND ila.lotNum = ad.toLotNum
LEFT JOIN dbo.BAS_LOCATION iz WITH (NOLOCK) ON iz.organizationId = ad.organizationId AND iz.warehouseId = ad.warehouseId AND iz.locationId = ad.locationId
WHERE ad.organizationId = 'ZXJT'
    AND ad.transactionType = 'IN'
    AND ad.status = '99'
    AND (ad.docResult = '2' OR ad.remark IS NOT NULL AND ad.remark <> '')
"""

SQL_INSPECTION_NORMAL = """
SELECT CONVERT(NVARCHAR(20), ad.orderDate, 23) AS order_date,
    ad.asnNo AS asn_no, ad.docType AS doc_type,
    iz.zoneName AS zone_name,
    bs.skudescr1 AS material_name, bs.alternate_sku3 AS material_code,
    ad.toLotNum AS production_batch, ila.lotAtt06 AS custom_batch,
    CASE WHEN ad.docResult = '1' THEN N'合格' WHEN ad.docResult = '2' THEN N'不合格' ELSE N'待检' END AS quality_status,
    ad.qty, ad.receiverId AS warehouse_keeper,
    ad.receiverId AS receiver, ad.reviewerId AS reviewer,
    ad.remark, 0 AS is_anomaly
FROM dbo.ACT_TRANSACTION_LOG ad WITH (NOLOCK)
LEFT JOIN dbo.BAS_SKU bs WITH (NOLOCK) ON bs.organizationId = ad.organizationId AND bs.customerId = ad.customerId AND bs.sku = ad.sku
LEFT JOIN dbo.INV_LOT_ATT ila WITH (NOLOCK) ON ila.organizationId = ad.organizationId AND ila.lotNum = ad.toLotNum
LEFT JOIN dbo.BAS_LOCATION iz WITH (NOLOCK) ON iz.organizationId = ad.organizationId AND iz.warehouseId = ad.warehouseId AND iz.locationId = ad.locationId
WHERE ad.organizationId = 'ZXJT'
    AND ad.transactionType = 'IN'
    AND ad.status = '99'
    AND ad.docResult = '1'
"""

SQL_SCADA_HOURLY_AGG = """
SELECT tagname, datetime, value
FROM History
WHERE wwTimeZone = 'China Standard Time'
  AND wwResolution = '60000'
  AND wwRetrievalMode = 'Cyclic'
  AND DateTime >= DATEADD(HOUR, -25, GETDATE())
ORDER BY tagname, datetime
"""


# ---------------------------------------------------------------------------
# ETL 管道类
# ---------------------------------------------------------------------------

class ETLPipeline:
    """DataAgentDW ETL 管道：从源库抽取 → 清洗转换 → 写入宽表"""

    def __init__(self):
        self._dw_config = self._read_dw_config()
        logger.info("[ETL] ETLPipeline 初始化完成")

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    @staticmethod
    def _read_dw_config() -> dict:
        """从环境变量读取 DataAgentDW 连接配置"""
        host = os.getenv("DB_DW_HOST")
        if not host:
            raise ValueError("缺少数据仓库配置: 请设置 DB_DW_HOST 等环境变量")

        return {
            "server": host,
            "port": int(os.getenv("DB_DW_PORT", "3306")),
            "user": os.getenv("DB_DW_USER", ""),
            "password": os.getenv("DB_DW_PASSWORD", ""),
            "database": os.getenv("DB_DW_DATABASE", DW_DB_NAME),
            "db_type": os.getenv("DB_DW_TYPE", "mysql"),
            "charset": "utf8",
            "timeout": 120,
            "login_timeout": 15,
        }

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @staticmethod
    def _get_source_connection(source_name: str):
        """获取源库 pymssql 连接"""
        cfg = get_db_config(source_name)
        if cfg is None:
            raise ValueError(f"源库 {source_name} 未配置，请检查环境变量")
        kwargs = cfg.to_pymssql_kwargs()
        # HISTORIAN (OSIsoft) 需要 TDS 7.0 协议
        if source_name.upper() == "HISTORIAN":
            kwargs["tds_version"] = "7.0"
        conn = pymssql.connect(**kwargs)
        return conn

    def _get_dw_connection(self):
        """获取 DataAgentDW 连接（支持 MySQL 和 SQL Server）"""
        if self._dw_config.get("db_type", "mysql") == "mysql":
            return pymysql.connect(
                host=self._dw_config["server"],
                port=self._dw_config["port"],
                user=self._dw_config["user"],
                password=self._dw_config["password"],
                database=self._dw_config["database"],
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
        else:
            return pymssql.connect(**self._dw_config)

    # ------------------------------------------------------------------
    # DW 写入操作（绕过 db_executor 的 SQL 安全校验）
    # ------------------------------------------------------------------

    def _truncate_table(self, table_name: str):
        """TRUNCATE 指定宽表"""
        conn = self._get_dw_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"TRUNCATE TABLE {table_name}")
            conn.commit()
            logger.info(f"[ETL] 已清空表 {table_name}")
        finally:
            conn.close()

    def _insert_rows(self, table_name: str, columns: list, rows: list):
        """批量 INSERT 行到宽表，每 BATCH_SIZE 条提交一次"""
        if not rows:
            logger.info(f"[ETL] {table_name}: 无数据需要插入")
            return

        col_list = ", ".join(f"`{c}`" for c in columns)  # MySQL uses backticks
        placeholders = ", ".join(["%s"] * len(columns))
        sql_template = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"

        conn = self._get_dw_connection()
        try:
            cursor = conn.cursor()
            inserted = 0
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i: i + BATCH_SIZE]
                batch_values = [self._prepare_row(row, columns) for row in batch]
                cursor.executemany(sql_template, batch_values)
                conn.commit()
                inserted += len(batch)
                logger.debug(
                    f"[ETL] {table_name}: 已插入 {inserted}/{len(rows)} 行"
                )
            logger.info(f"[ETL] {table_name}: 共插入 {inserted} 行")
        finally:
            conn.close()

    @staticmethod
    def _prepare_row(row: dict, columns: list) -> tuple:
        """
        将字典行转为按 columns 顺序的元组，
        同时处理 Decimal / datetime 等类型的序列化问题。
        """
        convert_row_types(row)
        result = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, bool):
                val = 1 if val else 0
            elif isinstance(val, Decimal):
                val = float(val)
            result.append(val)
        return tuple(result)

    # ------------------------------------------------------------------
    # 通用 ETL 任务运行器
    # ------------------------------------------------------------------

    def _run_etl_task(self, task_name: str, source_db: str,
                      source_sql: str, dw_table: str,
                      column_mapping: dict) -> dict:
        """
        通用 ETL 任务：TRUNCATE 目标表 → 从源库 SELECT → INSERT 到宽表

        Args:
            task_name: 任务名称（用于日志和状态记录）
            source_db: 源库连接名称（EAM/WMS_PROD/EKP/HISTORIAN）
            source_sql: 源库查询 SQL
            dw_table: 目标宽表名
            column_mapping: {源列名: 目标列名} 映射

        Returns:
            {"success": bool, "rows": int, "elapsed": float, "error": str|None}
        """
        t_start = time.time()
        logger.info(f"[ETL] 开始任务: {task_name}")
        try:
            # 1. 从源库读取
            conn = self._get_source_connection(source_db)
            try:
                cursor = conn.cursor(as_dict=True)
                cursor.execute(source_sql)
                rows = cursor.fetchall()
            finally:
                conn.close()

            logger.info(f"[ETL] {task_name}: 源库返回 {len(rows)} 行")

            # 2. 列名映射
            src_cols = list(column_mapping.keys())
            dst_cols = list(column_mapping.values())

            # 3. TRUNCATE + INSERT
            self._truncate_table(dw_table)
            self._insert_rows(dw_table, dst_cols, rows)

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"[ETL] {task_name}: 完成，耗时 {elapsed}s")
            return {
                "success": True,
                "rows": len(rows),
                "elapsed": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            logger.error(f"[ETL] {task_name}: 失败 - {e}")
            return {
                "success": False,
                "rows": 0,
                "elapsed": elapsed,
                "error": str(e)[:500],
            }

    # ------------------------------------------------------------------
    # 各 ETL 任务
    # ------------------------------------------------------------------

    def etl_workorder_detail(self) -> dict:
        """工单明细宽表 ETL"""
        return self._run_etl_task(
            task_name="dw_workorder_detail",
            source_db="EAM",
            source_sql=SQL_WORKORDER_DETAIL,
            dw_table="dw_workorder_detail",
            column_mapping={
                "RECCODE": "RECCODE",
                "RECFLOCODE": "RECFLOCODE",
                "FLODESC": "FLODESC",
                "FLOENTITYDESC": "FLOENTITYDESC",
                "CREATEDBY": "CREATEDBY",
                "EMPLOYEE_NAME": "EMPLOYEE_NAME",
                "DEPT_CODE": "DEPT_CODE",
                "DEPT_NAME": "DEPT_NAME",
                "CreateTime": "CreateTime",
                "ApprovalTime": "ApprovalTime",
                "ProcessHours": "ProcessHours",
                "ProcessDays": "ProcessDays",
                "ProcessTimeCategory": "ProcessTimeCategory",
                "RECFROMSTATUS": "RECFROMSTATUS",
            },
        )

    def etl_workorder_completion(self) -> dict:
        """工单完成情况宽表 ETL"""
        return self._run_etl_task(
            task_name="dw_workorder_completion",
            source_db="EAM",
            source_sql=SQL_WORKORDER_COMPLETION,
            dw_table="dw_workorder_completion",
            column_mapping={
                "wfscode": "wfscode",
                "WFSFLOCODE": "WFSFLOCODE",
                "flow_type": "flow_type",
                "completion_status": "completion_status",
                "LASTSAVED": "LASTSAVED",
            },
        )

    def etl_workorder_pending(self) -> dict:
        """待审批工单宽表 ETL"""
        return self._run_etl_task(
            task_name="dw_workorder_pending",
            source_db="EAM",
            source_sql=SQL_WORKORDER_PENDING,
            dw_table="dw_workorder_pending",
            column_mapping={
                "CREATEDBY": "CREATEDBY",
                "DEPT_CODE": "DEPT_CODE",
                "WFSSTATUS": "WFSSTATUS",
                "noddesc": "noddesc",
                "WFSENTITY": "WFSENTITY",
                "wfscode": "wfscode",
                "WFSFLOCODE": "WFSFLOCODE",
                "LASTSAVED": "LASTSAVED",
                "CurrentApproverID": "CurrentApproverID",
                "CurrentApprover": "CurrentApprover",
                "CurrentDepartment": "CurrentDepartment",
                "DaysSinceLastSave": "DaysSinceLastSave",
            },
        )

    def etl_department(self) -> dict:
        """部门维度表 ETL"""
        return self._run_etl_task(
            task_name="dw_department",
            source_db="EAM",
            source_sql=SQL_DEPARTMENT,
            dw_table="dw_department",
            column_mapping={
                "dept_code": "dept_code",
                "dept_name": "dept_name",
            },
        )

    def etl_flow_category(self) -> dict:
        """工单大类维度表 ETL"""
        return self._run_etl_task(
            task_name="dw_flow_category",
            source_db="EAM",
            source_sql=SQL_FLOW_CATEGORY,
            dw_table="dw_flow_category",
            column_mapping={
                "flow_category": "flow_category",
                "dept_code": "dept_code",
            },
        )

    def etl_flow_type(self) -> dict:
        """工单子类维度表 ETL"""
        return self._run_etl_task(
            task_name="dw_flow_type",
            source_db="EAM",
            source_sql=SQL_FLOW_TYPE,
            dw_table="dw_flow_type",
            column_mapping={
                "flow_type": "flow_type",
                "FLOENTITYDESC": "FLOENTITYDESC",
                "dept_code": "dept_code",
            },
        )

    def etl_inventory_by_zone(self) -> dict:
        """各库区库存汇总宽表 ETL（合并 ERP + WMS 数据源）"""
        t_start = time.time()
        task_name = "dw_inventory_by_zone"
        logger.info(f"[ETL] 开始任务: {task_name}")

        try:
            all_rows = []

            # 1. 从 EKP 读取 ERP 库存
            try:
                conn_ekp = self._get_source_connection("EKP")
                try:
                    cursor = conn_ekp.cursor(as_dict=True)
                    cursor.execute(SQL_INVENTORY_ERP)
                    erp_rows = cursor.fetchall()
                    all_rows.extend(erp_rows)
                    logger.info(f"[ETL] {task_name}: ERP 源返回 {len(erp_rows)} 行")
                finally:
                    conn_ekp.close()
            except Exception as e:
                logger.warning(f"[ETL] {task_name}: ERP 源查询失败 - {e}")

            # 2. 从 WMS_PROD 读取 WMS 库存
            try:
                conn_wms = self._get_source_connection("WMS_PROD")
                try:
                    cursor = conn_wms.cursor(as_dict=True)
                    cursor.execute(SQL_INVENTORY_WMS)
                    wms_rows = cursor.fetchall()
                    all_rows.extend(wms_rows)
                    logger.info(f"[ETL] {task_name}: WMS 源返回 {len(wms_rows)} 行")
                finally:
                    conn_wms.close()
            except Exception as e:
                logger.warning(f"[ETL] {task_name}: WMS 源查询失败 - {e}")

            # 3. 写入宽表
            column_mapping = {
                "source": "source",
                "zone_id": "zone_id",
                "zone_name": "zone_name",
                "qty": "qty",
            }
            dst_cols = list(column_mapping.values())

            self._truncate_table("dw_inventory_by_zone")
            self._insert_rows("dw_inventory_by_zone", dst_cols, all_rows)

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"[ETL] {task_name}: 完成，耗时 {elapsed}s")
            return {
                "success": True,
                "rows": len(all_rows),
                "elapsed": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            logger.error(f"[ETL] {task_name}: 失败 - {e}")
            return {
                "success": False,
                "rows": 0,
                "elapsed": elapsed,
                "error": str(e)[:500],
            }

    def etl_material_stagnant(self) -> dict:
        """呆滞物料宽表 ETL"""
        return self._run_etl_task(
            task_name="dw_material_stagnant",
            source_db="WMS_PROD",
            source_sql=SQL_MATERIAL_STAGNANT,
            dw_table="dw_material_stagnant",
            column_mapping={
                "material_type": "material_type",
                "sku_descr": "sku_descr",
                "sum_qty": "sum_qty",
                "lotAtt06": "lot_att06",
                "alternate_sku3": "alternate_sku3",
                "uomDescr": "uom_descr",
                "warehouseId": "warehouse_id",
                "warehouse_name": "warehouse_name",
                "days_no_out": "days_no_out",
            },
        )

    def etl_material_expiry(self) -> dict:
        """近效期物料宽表 ETL"""
        return self._run_etl_task(
            task_name="dw_material_expiry",
            source_db="WMS_PROD",
            source_sql=SQL_MATERIAL_EXPIRY,
            dw_table="dw_material_expiry",
            column_mapping={
                "material_type": "material_type",
                "expiry_date": "expiry_date",
                "lotAtt06": "lot_att06",
                "alternate_sku3": "alternate_sku3",
                "sku_descr": "sku_descr",
                "uomDescr": "uom_descr",
                "sum_qty": "sum_qty",
                "days_to_expiry": "days_to_expiry",
            },
        )

    def etl_inspection_record(self) -> dict:
        """验收记录宽表 ETL（合并异常 + 正常记录）"""
        t_start = time.time()
        task_name = "dw_inspection_record"
        logger.info(f"[ETL] 开始任务: {task_name}")

        try:
            all_rows = []

            # 1. 异常记录
            try:
                conn = self._get_source_connection("WMS_PROD")
                try:
                    cursor = conn.cursor(as_dict=True)
                    cursor.execute(SQL_INSPECTION_ANOMALY)
                    anomaly_rows = cursor.fetchall()
                    all_rows.extend(anomaly_rows)
                    logger.info(f"[ETL] {task_name}: 异常记录 {len(anomaly_rows)} 行")
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"[ETL] {task_name}: 异常记录查询失败 - {e}")

            # 2. 正常记录
            try:
                conn = self._get_source_connection("WMS_PROD")
                try:
                    cursor = conn.cursor(as_dict=True)
                    cursor.execute(SQL_INSPECTION_NORMAL)
                    normal_rows = cursor.fetchall()
                    all_rows.extend(normal_rows)
                    logger.info(f"[ETL] {task_name}: 正常记录 {len(normal_rows)} 行")
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"[ETL] {task_name}: 正常记录查询失败 - {e}")

            # 3. 写入宽表
            column_mapping = {
                "order_date": "order_date",
                "asn_no": "asn_no",
                "doc_type": "doc_type",
                "zone_name": "zone_name",
                "material_name": "material_name",
                "material_code": "material_code",
                "production_batch": "production_batch",
                "custom_batch": "custom_batch",
                "quality_status": "quality_status",
                "qty": "quantity",
                "warehouse_keeper": "warehouse_keeper",
                "receiver": "receiver",
                "reviewer": "reviewer",
                "remark": "remark",
                "is_anomaly": "is_anomaly",
            }
            dst_cols = list(column_mapping.values())

            self._truncate_table("dw_inspection_record")
            self._insert_rows("dw_inspection_record", dst_cols, all_rows)

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"[ETL] {task_name}: 完成，耗时 {elapsed}s")
            return {
                "success": True,
                "rows": len(all_rows),
                "elapsed": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            logger.error(f"[ETL] {task_name}: 失败 - {e}")
            return {
                "success": False,
                "rows": 0,
                "elapsed": elapsed,
                "error": str(e)[:500],
            }

    def etl_scada_hourly_agg(self) -> dict:
        """SCADA 小时聚合宽表 ETL

        从 Historian 读取原始时序数据，在 Python 中完成：
        1. tagname → room_name / measure_type 映射（借助 FMCSDeviceRegistry）
        2. 按小时聚合 avg / max / min / count
        3. 写入 dw_scada_hourly_agg
        """
        t_start = time.time()
        task_name = "dw_scada_hourly_agg"
        logger.info(f"[ETL] 开始任务: {task_name}")

        try:
            # 1. 从 Historian 读取原始数据
            conn = self._get_source_connection("HISTORIAN")
            try:
                cursor = conn.cursor(as_dict=True)
                cursor.execute(SQL_SCADA_HOURLY_AGG)
                raw_rows = cursor.fetchall()
            finally:
                conn.close()

            logger.info(f"[ETL] {task_name}: Historian 返回 {len(raw_rows)} 行原始数据")

            if not raw_rows:
                self._truncate_table("dw_scada_hourly_agg")
                elapsed = round(time.time() - t_start, 2)
                return {"success": True, "rows": 0, "elapsed": elapsed, "error": None}

            # 2. 构建 tagname → (room_name, measure_type) 映射
            tag_meta = self._build_tagname_meta()

            # 3. 在 Python 中按 (tagname, hour) 聚合
            from collections import defaultdict
            hourly_buckets: dict = defaultdict(list)
            for row in raw_rows:
                tagname = row.get("tagname", "")
                dt_val = row.get("datetime")
                value = row.get("value")

                if not tagname or dt_val is None or value is None:
                    continue

                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                # 提取小时键
                if isinstance(dt_val, datetime):
                    hour_key = dt_val.replace(minute=0, second=0, microsecond=0)
                else:
                    try:
                        parsed = datetime.strptime(str(dt_val).split('.')[0], "%Y-%m-%d %H:%M:%S")
                        hour_key = parsed.replace(minute=0, second=0, microsecond=0)
                    except ValueError:
                        continue

                bucket_key = (tagname, hour_key)
                hourly_buckets[bucket_key].append(value)

            # 4. 生成聚合行
            agg_rows = []
            for (tagname, hour_key), values in hourly_buckets.items():
                meta = tag_meta.get(tagname, {})
                agg_rows.append({
                    "tag_name": tagname,
                    "room_name": meta.get("room_name", ""),
                    "measure_type": meta.get("measure_type", ""),
                    "stat_hour": hour_key.strftime("%Y-%m-%d %H:00:00"),
                    "avg_value": round(sum(values) / len(values), 4),
                    "max_value": round(max(values), 4),
                    "min_value": round(min(values), 4),
                    "sample_count": len(values),
                })

            logger.info(f"[ETL] {task_name}: 聚合为 {len(agg_rows)} 行小时数据")

            # 5. TRUNCATE + INSERT
            column_mapping = {
                "tag_name": "tag_name",
                "room_name": "room_name",
                "measure_type": "measure_type",
                "stat_hour": "stat_hour",
                "avg_value": "avg_value",
                "max_value": "max_value",
                "min_value": "min_value",
                "sample_count": "sample_count",
            }
            dst_cols = list(column_mapping.values())

            self._truncate_table("dw_scada_hourly_agg")
            self._insert_rows("dw_scada_hourly_agg", dst_cols, agg_rows)

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"[ETL] {task_name}: 完成，耗时 {elapsed}s")
            return {"success": True, "rows": len(agg_rows), "elapsed": elapsed, "error": None}

        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            logger.error(f"[ETL] {task_name}: 失败 - {e}")
            return {"success": False, "rows": 0, "elapsed": elapsed, "error": str(e)[:500]}

    @staticmethod
    def _build_tagname_meta() -> dict:
        """从 FMCSDeviceRegistry 构建 tagname → {room_name, measure_type} 映射"""
        try:
            from utils.fmcs_registry import FMCSDeviceRegistry, MEASUREMENT_TYPES
            registry = FMCSDeviceRegistry()
            registry._ensure_loaded()
            meta = {}
            for device in registry.devices:
                tagname = device.get("tagname", "")
                mt = device.get("measure_type", "")
                measure_info = MEASUREMENT_TYPES.get(mt, MEASUREMENT_TYPES.get("PV", {}))
                meta[tagname] = {
                    "room_name": device.get("room_name", ""),
                    "measure_type": measure_info.get("label", mt),
                }
            logger.info(f"[ETL] tagname 元数据: {len(meta)} 条")
            return meta
        except Exception as e:
            logger.warning(f"[ETL] 加载 FMCSDeviceRegistry 失败，room_name/measure_type 将为空: {e}")
            return {}

    def etl_scada_threshold_event(self) -> dict:
        """
        SCADA 阈值超标事件宽表 ETL

        从 dw_scada_hourly_agg 中检测超标小时段，
        并合并连续超标小时为单个事件。
        注意：基于小时聚合数据检测，精度为小时级，标记为估算。
        后续可增强为基于原始采样数据的精确检测。
        """
        t_start = time.time()
        task_name = "dw_scada_threshold_event"
        logger.info(f"[ETL] 开始任务: {task_name}")

        try:
            # 从已聚合的小时数据中检测超标事件
            threshold_sql = """
            SELECT tag_name, room_name, measure_type,
                stat_hour, avg_value, max_value, min_value, sample_count
            FROM dw_scada_hourly_agg
            WHERE (measure_type = '温度' AND max_value > 25.0)
               OR (measure_type = '湿度' AND max_value > 65.0)
               OR (measure_type = '压差' AND min_value < 10.0)
            ORDER BY tag_name, stat_hour
            """

            conn = self._get_dw_connection()
            try:
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute(threshold_sql)
                raw_events = cursor.fetchall()
            finally:
                conn.close()

            logger.info(f"[ETL] {task_name}: 检测到 {len(raw_events)} 个超标小时段")

            # 合并连续超标小时为单个事件
            events = self._merge_continuous_events(raw_events)

            logger.info(f"[ETL] {task_name}: 合并为 {len(events)} 个超标事件")

            column_mapping = {
                "tag_name": "tag_name",
                "room_name": "room_name",
                "measure_type": "measure_type",
                "threshold_value": "threshold_value",
                "threshold_operator": "threshold_operator",
                "start_time": "start_time",
                "end_time": "end_time",
                "duration_minutes": "duration_minutes",
                "max_value": "max_value",
                "avg_value": "avg_value",
                "stat_date": "stat_date",
                "is_estimated": "is_estimated",
            }
            dst_cols = list(column_mapping.values())

            self._truncate_table("dw_scada_threshold_event")
            self._insert_rows("dw_scada_threshold_event", dst_cols, events)

            elapsed = round(time.time() - t_start, 2)
            logger.info(f"[ETL] {task_name}: 完成，耗时 {elapsed}s")
            return {
                "success": True,
                "rows": len(events),
                "elapsed": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            logger.error(f"[ETL] {task_name}: 失败 - {e}")
            return {
                "success": False,
                "rows": 0,
                "elapsed": elapsed,
                "error": str(e)[:500],
            }

    @staticmethod
    def _merge_continuous_events(raw_events: list) -> list:
        """将连续超标的小时段合并为单个事件"""
        if not raw_events:
            return []

        # 阈值配置
        threshold_config = {
            '温度': {'value': 25.0, 'operator': '>'},
            '湿度': {'value': 65.0, 'operator': '>'},
            '压差': {'value': 10.0, 'operator': '<'},
        }

        # 按 (tag_name, measure_type) 分组
        from collections import defaultdict
        groups = defaultdict(list)
        for row in raw_events:
            key = (row.get('tag_name', ''), row.get('measure_type', ''))
            groups[key].append(row)

        events = []
        for (tag_name, measure_type), hours in groups.items():
            cfg = threshold_config.get(measure_type, {'value': 999.0, 'operator': '>'})
            # 按时间排序
            hours.sort(key=lambda h: h.get('stat_hour', ''))

            # 合并连续小时
            current_start = hours[0].get('stat_hour')
            current_end = hours[0].get('stat_hour')
            max_val = hours[0].get('max_value', 0)
            sum_avg = hours[0].get('avg_value', 0)
            count = 1

            for i in range(1, len(hours)):
                h = hours[i]
                h_stat = h.get('stat_hour')
                # 判断是否连续（间隔1小时以内）
                if current_end and h_stat:
                    try:
                        from datetime import datetime, timedelta
                        if isinstance(current_end, str):
                            end_dt = datetime.strptime(current_end.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        else:
                            end_dt = current_end
                        if isinstance(h_stat, str):
                            h_dt = datetime.strptime(h_stat.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        else:
                            h_dt = h_stat
                        if (h_dt - end_dt).total_seconds() <= 3600:
                            # 连续，合并
                            current_end = h_stat
                            max_val = max(max_val, h.get('max_value', 0) or 0)
                            sum_avg += h.get('avg_value', 0) or 0
                            count += 1
                            continue
                    except (ValueError, TypeError):
                        pass

                # 不连续，保存当前事件
                duration = count * 60
                stat_date = current_start
                if isinstance(current_start, datetime):
                    stat_date = current_start.strftime("%Y-%m-%d")
                elif isinstance(current_start, str):
                    stat_date = current_start[:10]

                events.append({
                    "tag_name": tag_name,
                    "room_name": hours[0].get('room_name', ''),
                    "measure_type": measure_type,
                    "threshold_value": cfg['value'],
                    "threshold_operator": cfg['operator'],
                    "start_time": str(current_start),
                    "end_time": str(current_end),
                    "duration_minutes": duration,
                    "max_value": round(float(max_val), 4) if max_val else None,
                    "avg_value": round(float(sum_avg / count), 4) if count else None,
                    "stat_date": stat_date,
                    "is_estimated": 1,
                })

                # 开始新事件
                current_start = h_stat
                current_end = h_stat
                max_val = h.get('max_value', 0)
                sum_avg = h.get('avg_value', 0) or 0
                count = 1

            # 保存最后一个事件
            duration = count * 60
            stat_date = current_start
            if isinstance(current_start, datetime):
                stat_date = current_start.strftime("%Y-%m-%d")
            elif isinstance(current_start, str):
                stat_date = current_start[:10]

            events.append({
                "tag_name": tag_name,
                "room_name": hours[0].get('room_name', ''),
                "measure_type": measure_type,
                "threshold_value": cfg['value'],
                "threshold_operator": cfg['operator'],
                "start_time": str(current_start),
                "end_time": str(current_end),
                "duration_minutes": duration,
                "max_value": round(float(max_val), 4) if max_val else None,
                "avg_value": round(float(sum_avg / count), 4) if count else None,
                "stat_date": stat_date,
                "is_estimated": 1,
            })

        return events

    # ------------------------------------------------------------------
    # 全量运行
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """
        顺序执行所有 ETL 任务，记录每个任务的执行状态。

        Returns:
            {task_name: {success, rows, elapsed, error}, ...}
        """
        global _etl_status
        logger.info("[ETL] ========== 开始全量 ETL ==========")
        t_total = time.time()

        tasks = [
            # EAM 工单域
            ("dw_department", self.etl_department),
            ("dw_flow_category", self.etl_flow_category),
            ("dw_flow_type", self.etl_flow_type),
            ("dw_workorder_detail", self.etl_workorder_detail),
            ("dw_workorder_completion", self.etl_workorder_completion),
            ("dw_workorder_pending", self.etl_workorder_pending),
            # WMS 域（暂未建表，跳过）
            # ("dw_inventory_by_zone", self.etl_inventory_by_zone),
            # ("dw_material_stagnant", self.etl_material_stagnant),
            # ("dw_material_expiry", self.etl_material_expiry),
            # ("dw_inspection_record", self.etl_inspection_record),
            # SCADA 域
            ("dw_scada_hourly_agg", self.etl_scada_hourly_agg),
            ("dw_scada_threshold_event", self.etl_scada_threshold_event),
        ]

        results = {}
        for task_name, task_func in tasks:
            result = task_func()
            results[task_name] = result

        total_elapsed = round(time.time() - t_total, 2)
        all_success = all(r["success"] for r in results.values())

        _etl_status = {
            "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_success": all_success,
            "total_elapsed": total_elapsed,
            "tasks": results,
        }

        success_count = sum(1 for r in results.values() if r["success"])
        fail_count = len(results) - success_count
        logger.info(
            f"[ETL] ========== 全量 ETL 完成 | "
            f"成功={success_count} 失败={fail_count} 总耗时={total_elapsed}s =========="
        )

        return results

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @staticmethod
    def get_status() -> dict:
        """返回最近一次 ETL 运行状态"""
        return dict(_etl_status)
