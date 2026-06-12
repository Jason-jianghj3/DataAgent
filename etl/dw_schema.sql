-- ============================================================
-- DataAgentDW 数据仓库 DDL (MySQL 版本)
-- 所有宽表以 dw_ 为前缀，供 LLM 生成简单 SELECT 查询
-- ============================================================

-- 1. 工单明细宽表
CREATE TABLE IF NOT EXISTS dw_workorder_detail (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    RECCODE VARCHAR(100),
    RECFLOCODE VARCHAR(100),
    FLODESC VARCHAR(200),
    FLOENTITYDESC VARCHAR(200),
    CREATEDBY VARCHAR(100),
    EMPLOYEE_NAME VARCHAR(100),
    DEPT_CODE VARCHAR(50),
    DEPT_NAME VARCHAR(200),
    CreateTime DATETIME,
    ApprovalTime DATETIME,
    ProcessHours DECIMAL(18,2),
    ProcessDays DECIMAL(18,2),
    ProcessTimeCategory VARCHAR(50),
    RECFROMSTATUS VARCHAR(200),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dept_code (DEPT_CODE),
    INDEX idx_approval_time (ApprovalTime),
    INDEX idx_create_time (CreateTime)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. 工单完成情况宽表
CREATE TABLE IF NOT EXISTS dw_workorder_completion (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    wfscode VARCHAR(100),
    WFSFLOCODE VARCHAR(100),
    flow_type VARCHAR(200),
    completion_status VARCHAR(20),
    LASTSAVED DATETIME,
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_completion_status (completion_status)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. 待审批工单宽表
CREATE TABLE IF NOT EXISTS dw_workorder_pending (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    CREATEDBY VARCHAR(100),
    DEPT_CODE VARCHAR(50),
    WFSSTATUS VARCHAR(100),
    noddesc VARCHAR(200),
    WFSENTITY VARCHAR(200),
    wfscode VARCHAR(100),
    WFSFLOCODE VARCHAR(200),
    LASTSAVED DATETIME,
    CurrentApproverID TEXT,
    CurrentApprover TEXT,
    CurrentDepartment TEXT,
    DaysSinceLastSave DECIMAL(18,4),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dept_code (DEPT_CODE)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. 部门维度表
CREATE TABLE IF NOT EXISTS dw_department (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dept_code VARCHAR(50) NOT NULL,
    dept_name VARCHAR(200),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. 工单大类维度表
CREATE TABLE IF NOT EXISTS dw_flow_category (
    id INT AUTO_INCREMENT PRIMARY KEY,
    flow_category VARCHAR(200),
    dept_code VARCHAR(50),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. 工单子类维度表
CREATE TABLE IF NOT EXISTS dw_flow_type (
    id INT AUTO_INCREMENT PRIMARY KEY,
    flow_type VARCHAR(200),
    FLOENTITYDESC VARCHAR(200),
    dept_code VARCHAR(50),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. 各库区库存汇总宽表
CREATE TABLE IF NOT EXISTS dw_inventory_by_zone (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(20),
    zone_id VARCHAR(50),
    zone_name VARCHAR(100),
    qty DECIMAL(18,3),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_source (source)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. 呆滞物料宽表
CREATE TABLE IF NOT EXISTS dw_material_stagnant (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    material_type VARCHAR(100),
    sku_descr VARCHAR(200),
    sum_qty DECIMAL(18,3),
    lot_att06 VARCHAR(100),
    alternate_sku3 VARCHAR(200),
    uom_descr VARCHAR(100),
    warehouse_id VARCHAR(50),
    warehouse_name VARCHAR(200),
    days_no_out INT,
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. 近效期物料宽表
CREATE TABLE IF NOT EXISTS dw_material_expiry (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    material_type VARCHAR(100),
    expiry_date VARCHAR(100),
    lot_att06 VARCHAR(100),
    alternate_sku3 VARCHAR(200),
    sku_descr VARCHAR(200),
    uom_descr VARCHAR(100),
    sum_qty DECIMAL(18,3),
    days_to_expiry INT,
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10. 验收记录宽表
CREATE TABLE IF NOT EXISTS dw_inspection_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_date VARCHAR(20),
    asn_no VARCHAR(100),
    doc_type VARCHAR(100),
    zone_name VARCHAR(200),
    material_name VARCHAR(200),
    material_code VARCHAR(100),
    production_batch VARCHAR(200),
    custom_batch VARCHAR(100),
    quality_status VARCHAR(100),
    quantity DECIMAL(18,3),
    warehouse_keeper VARCHAR(100),
    receiver VARCHAR(100),
    reviewer VARCHAR(100),
    remark TEXT,
    is_anomaly TINYINT(1),
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_anomaly (is_anomaly)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 11. SCADA小时聚合宽表
CREATE TABLE IF NOT EXISTS dw_scada_hourly_agg (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    tag_name VARCHAR(200),
    room_name VARCHAR(100),
    measure_type VARCHAR(50),
    stat_hour DATETIME,
    avg_value DECIMAL(18,4),
    max_value DECIMAL(18,4),
    min_value DECIMAL(18,4),
    sample_count INT,
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_room_name (room_name),
    INDEX idx_stat_hour (stat_hour)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 12. SCADA阈值超标事件宽表
CREATE TABLE IF NOT EXISTS dw_scada_threshold_event (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    tag_name VARCHAR(200),
    room_name VARCHAR(100),
    measure_type VARCHAR(50),
    threshold_value DECIMAL(18,4),
    threshold_operator VARCHAR(10),
    start_time DATETIME,
    end_time DATETIME,
    duration_minutes INT,
    max_value DECIMAL(18,4),
    avg_value DECIMAL(18,4),
    stat_date DATE,
    etl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_stat_date (stat_date),
    INDEX idx_room_measure (room_name, measure_type)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
