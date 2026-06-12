"""初始化数仓库 - 创建数据库和宽表"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymssql
from utils.db_config import get_db_config
from utils.logger import logger


def create_dw_database():
    """创建DataAgentDW数据库（如果不存在）"""
    cfg = get_db_config('EAM')
    if not cfg:
        logger.error("[DW Init] 无法获取EAM配置")
        return False

    try:
        conn = pymssql.connect(
            server=cfg.host, port=cfg.port,
            user=cfg.user, password=cfg.password,
            database='master', autocommit=True
        )
        cursor = conn.cursor()
        cursor.execute("SELECT database_id FROM sys.databases WHERE name = 'DataAgentDW'")
        row = cursor.fetchone()
        if row:
            logger.info("[DW Init] 数据库 DataAgentDW 已存在")
        else:
            cursor.execute('CREATE DATABASE DataAgentDW')
            conn.commit()
            logger.info("[DW Init] 数据库 DataAgentDW 创建成功")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"[DW Init] 创建数据库失败: {e}")
        return False


def create_dw_tables():
    """创建数仓宽表"""
    from utils.db_config import get_db_config as _cfg
    dw_cfg = _cfg('DW')
    if not dw_cfg:
        # 降级使用EAM配置连接DataAgentDW
        eam_cfg = _cfg('EAM')
        if not eam_cfg:
            logger.error("[DW Init] 无法获取数据库配置")
            return False
        server, port, user, password = eam_cfg.host, eam_cfg.port, eam_cfg.user, eam_cfg.password
    else:
        server, port, user, password = dw_cfg.host, dw_cfg.port, dw_cfg.user, dw_cfg.password

    try:
        conn = pymssql.connect(
            server=server, port=port,
            user=user, password=password,
            database='DataAgentDW'
        )
        cursor = conn.cursor()

        # 定义所有宽表的DDL
        tables = {
            'dw_workorder_detail': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_workorder_detail')
                CREATE TABLE dw_workorder_detail (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    RECCODE NVARCHAR(100),
                    RECFLOCODE NVARCHAR(100),
                    FLODESC NVARCHAR(200),
                    FLOENTITYDESC NVARCHAR(200),
                    CREATEDBY NVARCHAR(100),
                    EMPLOYEE_NAME NVARCHAR(100),
                    DEPT_CODE NVARCHAR(50),
                    DEPT_NAME NVARCHAR(200),
                    CreateTime DATETIME,
                    ApprovalTime DATETIME,
                    ProcessHours DECIMAL(18,2),
                    ProcessDays DECIMAL(18,2),
                    ProcessTimeCategory NVARCHAR(50),
                    RECFROMSTATUS NVARCHAR(200),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_workorder_completion': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_workorder_completion')
                CREATE TABLE dw_workorder_completion (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    wfscode NVARCHAR(100),
                    WFSFLOCODE NVARCHAR(100),
                    flow_type NVARCHAR(200),
                    completion_status NVARCHAR(20),
                    LASTSAVED DATETIME,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_workorder_pending': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_workorder_pending')
                CREATE TABLE dw_workorder_pending (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    CREATEDBY NVARCHAR(100),
                    DEPT_CODE NVARCHAR(50),
                    WFSSTATUS NVARCHAR(100),
                    noddesc NVARCHAR(200),
                    WFSENTITY NVARCHAR(200),
                    wfscode NVARCHAR(100),
                    WFSFLOCODE NVARCHAR(200),
                    LASTSAVED DATETIME,
                    CurrentApproverID NVARCHAR(100),
                    CurrentApprover NVARCHAR(100),
                    CurrentDepartment NVARCHAR(200),
                    DaysSinceLastSave DECIMAL(18,4),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_department': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_department')
                CREATE TABLE dw_department (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    dept_code NVARCHAR(50) NOT NULL,
                    dept_name NVARCHAR(200),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_flow_category': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_flow_category')
                CREATE TABLE dw_flow_category (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    flow_category NVARCHAR(200),
                    dept_code NVARCHAR(50),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_flow_type': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_flow_type')
                CREATE TABLE dw_flow_type (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    flow_type NVARCHAR(200),
                    FLOENTITYDESC NVARCHAR(200),
                    dept_code NVARCHAR(50),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_inventory_by_zone': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_inventory_by_zone')
                CREATE TABLE dw_inventory_by_zone (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    source NVARCHAR(20),
                    zone_id NVARCHAR(50),
                    zone_name NVARCHAR(100),
                    qty DECIMAL(18,3),
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_material_stagnant': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_material_stagnant')
                CREATE TABLE dw_material_stagnant (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    material_type NVARCHAR(100),
                    sku_descr NVARCHAR(200),
                    sum_qty DECIMAL(18,3),
                    lot_att06 NVARCHAR(100),
                    alternate_sku3 NVARCHAR(200),
                    uom_descr NVARCHAR(100),
                    warehouse_id NVARCHAR(50),
                    warehouse_name NVARCHAR(200),
                    days_no_out INT,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_material_expiry': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_material_expiry')
                CREATE TABLE dw_material_expiry (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    material_type NVARCHAR(100),
                    expiry_date NVARCHAR(100),
                    lot_att06 NVARCHAR(100),
                    alternate_sku3 NVARCHAR(200),
                    sku_descr NVARCHAR(200),
                    uom_descr NVARCHAR(100),
                    sum_qty DECIMAL(18,3),
                    days_to_expiry INT,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_inspection_record': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_inspection_record')
                CREATE TABLE dw_inspection_record (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    order_date NVARCHAR(20),
                    asn_no NVARCHAR(100),
                    doc_type NVARCHAR(100),
                    zone_name NVARCHAR(200),
                    material_name NVARCHAR(200),
                    material_code NVARCHAR(100),
                    production_batch NVARCHAR(200),
                    custom_batch NVARCHAR(100),
                    quality_status NVARCHAR(100),
                    quantity DECIMAL(18,3),
                    warehouse_keeper NVARCHAR(100),
                    receiver NVARCHAR(100),
                    reviewer NVARCHAR(100),
                    remark NVARCHAR(4000),
                    is_anomaly BIT,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_scada_hourly_agg': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_scada_hourly_agg')
                CREATE TABLE dw_scada_hourly_agg (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    tag_name NVARCHAR(200),
                    room_name NVARCHAR(100),
                    measure_type NVARCHAR(50),
                    stat_hour DATETIME,
                    avg_value DECIMAL(18,4),
                    max_value DECIMAL(18,4),
                    min_value DECIMAL(18,4),
                    sample_count INT,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
            'dw_scada_threshold_event': '''
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dw_scada_threshold_event')
                CREATE TABLE dw_scada_threshold_event (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    tag_name NVARCHAR(200),
                    room_name NVARCHAR(100),
                    measure_type NVARCHAR(50),
                    threshold_value DECIMAL(18,4),
                    threshold_operator NVARCHAR(10),
                    start_time DATETIME,
                    end_time DATETIME,
                    duration_minutes INT,
                    max_value DECIMAL(18,4),
                    avg_value DECIMAL(18,4),
                    stat_date DATE,
                    etl_time DATETIME DEFAULT GETDATE()
                )
            ''',
        }

        created = 0
        for table_name, ddl in tables.items():
            try:
                cursor.execute(ddl)
                conn.commit()
                created += 1
                logger.info(f"[DW Init] 表 {table_name} 创建/验证成功")
            except Exception as e:
                logger.error(f"[DW Init] 表 {table_name} 创建失败: {e}")

        conn.close()
        logger.info(f"[DW Init] 完成: {created}/{len(tables)} 张宽表已创建")
        return created == len(tables)

    except Exception as e:
        logger.error(f"[DW Init] 创建宽表失败: {e}")
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("  初始化数仓库 DataAgentDW")
    print("=" * 50)

    # 步骤1: 创建数据库
    if create_dw_database():
        print("[OK] 数据库创建成功")
    else:
        print("[FAIL] 数据库创建失败")
        exit(1)

    # 步骤2: 创建宽表
    if create_dw_tables():
        print("[OK] 宽表创建成功")
    else:
        print("[FAIL] 宽表创建失败")
        exit(1)

    # 步骤3: 运行首次ETL
    print("\n开始首次ETL数据加载...")
    try:
        from etl.etl_pipeline import ETLPipeline
        pipeline = ETLPipeline()
        results = pipeline.run_all()
        success_count = sum(1 for r in results.values() if r.get('success'))
        print(f"\n[ETL] 完成: {success_count}/{len(results)} 个任务成功")
        for task_name, result in results.items():
            status = "OK" if result.get('success') else "FAIL"
            rows = result.get('rows', 0)
            error = result.get('error', '')
            print(f"  [{status}] {task_name}: {rows} rows" + (f" ({error})" if error else ""))
    except Exception as e:
        print(f"[FAIL] ETL执行失败: {e}")

    print("\n数仓库初始化完成！")
