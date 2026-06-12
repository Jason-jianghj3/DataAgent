"""
时序数据智能分析服务 - 基于ADTK + Pandas resample

支持多场景分析：阈值超标、趋势分析、聚合统计、多设备对比、异常检测
适用于SCADA时序数据和工单效能数据的通用分析引擎
"""
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from adtk.data import validate_series
from adtk.detector import ThresholdAD, PersistAD, LevelShiftAD, VolatilityShiftAD, SeasonalAD
from adtk.transformer import RollingAggregate, DoubleRollingAggregate

from utils.logger import logger


def _parse_dt(dt_str: str) -> datetime:
    """安全解析datetime字符串，兼容带小数秒的格式如 '2025-05-21 10:30:00.0000000'"""
    clean = str(dt_str).split('.')[0].strip()
    return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")


def _parse_dt_hour(dt_str: str) -> datetime:
    """解析小时级datetime字符串如 '2025-05-21 10:00'"""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:00")


def _parse_dt_date(dt_str: str) -> datetime:
    """解析日期字符串如 '2025-05-21'"""
    return datetime.strptime(dt_str, "%Y-%m-%d")


def _to_series(data: List[Dict], freq: str = None) -> pd.Series:
    """将 [{datetime, value}, ...] 转为 pandas Series（DatetimeIndex）"""
    if not data:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([_parse_dt(p["datetime"]) for p in data])
    values = [p["value"] for p in data]
    s = pd.Series(values, index=index)
    s = s.sort_index()
    if freq:
        s = s.asfreq(freq)
    return s


def _to_multi_series(multi_data: Dict[str, List[Dict]]) -> Dict[str, pd.Series]:
    """将 {tagname: [{datetime, value}, ...]} 转为 {tagname: pd.Series}"""
    return {tn: _to_series(data) for tn, data in multi_data.items()}


class TimeSeriesAnalyzer:
    """时序数据智能分析引擎（基于ADTK + Pandas）

    适用于SCADA时序数据和工单效能数据的通用分析。
    分析类型：
    - raw: 原始时序数据查看
    - threshold: 阈值超标分析（超标时段、持续时长）
    - comparison: 多序列对比分析
    - trend: 趋势分析（变化率、阶跃检测）
    - aggregation: 聚合统计（小时/天均值）
    - anomaly: 异常检测（持续异常、阶跃突变、波动异常、季节性异常）
    """

    ANALYSIS_RAW = "raw"
    ANALYSIS_THRESHOLD = "threshold"
    ANALYSIS_COMPARISON = "comparison"
    ANALYSIS_TREND = "trend"
    ANALYSIS_AGGREGATION = "aggregation"
    ANALYSIS_ANOMALY = "anomaly"

    def __init__(self, historian_config=None):
        self._config = historian_config
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            from utils.fmcs_registry import FMCSDeviceRegistry
            self._registry = FMCSDeviceRegistry()
        return self._registry

    # ==================== 数据获取 ====================

    def fetch_timeseries(
        self,
        tagnames: List[str],
        start_date: str,
        end_date: str,
        resolution: int = 60000,
    ) -> Dict[str, List[Dict]]:
        """从Historian查询多个TagName的时序数据"""
        if not self._config:
            return {}

        import pymssql

        result = {}
        kwargs = self._config.to_pymssql_kwargs()
        kwargs['tds_version'] = '7.0'

        try:
            conn = pymssql.connect(**kwargs)
            cursor = conn.cursor()

            for tagname in tagnames:
                sql = """
                SELECT datetime, value, tagname FROM History
                WHERE TagName = %s
                  AND wwTimeZone = 'China Standard Time'
                  AND wwResolution = %s
                  AND wwRetrievalMode = 'Cyclic'
                  AND DateTime >= %s AND DateTime <= %s
                ORDER BY datetime ASC
                """
                try:
                    cursor.execute(sql, (tagname, str(resolution), start_date, end_date))
                    rows = cursor.fetchall()
                    data = []
                    for row in rows:
                        dt = row[0]
                        val = row[1]
                        if isinstance(dt, datetime):
                            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            dt_str = str(dt).split('.')[0]
                        try:
                            val = round(float(val), 4)
                        except (ValueError, TypeError):
                            continue
                        data.append({"datetime": dt_str, "value": val})
                    result[tagname] = data
                    logger.info(f"[TimeSeriesAnalyzer] {tagname}: 获取{len(data)}个数据点")
                except Exception as e:
                    logger.error(f"[TimeSeriesAnalyzer] 查询{tagname}失败: {e}")
                    result[tagname] = []

            conn.close()
        except Exception as e:
            logger.error(f"[TimeSeriesAnalyzer] Historian连接失败: {e}")

        return result

    # ==================== 核心分析方法 ====================

    def analyze_threshold(
        self,
        data: List[Dict],
        threshold: float,
        operator: str = ">",
    ) -> Dict:
        """阈值分析：用ADTK ThresholdAD检测超标时段及持续时长

        Args:
            data: [{datetime, value}, ...]
            threshold: 阈值
            operator: ">" 超过 / "<" 低于
        """
        if not data:
            return {"periods": [], "total_duration_min": 0, "max_duration_min": 0}

        try:
            s = _to_series(data)
            s = validate_series(s)

            if operator == "<":
                detector = ThresholdAD(low=threshold)
            else:
                detector = ThresholdAD(high=threshold)

            anomalies = detector.detect(s, return_list=True)

            # 将ADTK返回的事件列表转为统一格式
            periods = []
            for event in anomalies:
                if isinstance(event, tuple):
                    start, end = event
                    duration_min = (end - start).total_seconds() / 60
                    if duration_min == 0:
                        duration_min = 1  # 至少1分钟（单点超标）
                    # 获取该时段内的最大值
                    mask = (s.index >= start) & (s.index <= end)
                    period_values = s[mask]
                    max_val = round(float(period_values.max()), 2) if len(period_values) > 0 else None
                    periods.append({
                        "start": start.strftime("%Y-%m-%d %H:%M"),
                        "end": end.strftime("%Y-%m-%d %H:%M"),
                        "duration_min": round(duration_min, 1),
                        "max_value": max_val,
                    })
                elif isinstance(event, pd.Timestamp):
                    # 单点异常
                    val = s.get(event, None)
                    periods.append({
                        "start": event.strftime("%Y-%m-%d %H:%M"),
                        "end": event.strftime("%Y-%m-%d %H:%M"),
                        "duration_min": 1,
                        "max_value": round(float(val), 2) if pd.notna(val) else None,
                    })

            total_duration = sum(p["duration_min"] for p in periods)
            max_duration = max((p["duration_min"] for p in periods), default=0)

            return {
                "periods": periods,
                "total_duration_min": round(total_duration, 1),
                "max_duration_min": round(max_duration, 1),
                "threshold": threshold,
                "operator": operator,
                "exceed_count": len(periods),
            }

        except Exception as e:
            logger.warning(f"[TimeSeriesAnalyzer] ADTK阈值分析失败，降级到手动计算: {e}")
            return self._threshold_fallback(data, threshold, operator)

    def _threshold_fallback(self, data: List[Dict], threshold: float, operator: str) -> Dict:
        """阈值分析降级：手动遍历计算"""
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
        }
        op_func = ops.get(operator, ops[">"])

        periods = []
        current_start = None
        prev_dt = None

        for point in data:
            dt_str = point["datetime"]
            val = point["value"]
            dt = _parse_dt(dt_str)

            if op_func(val, threshold):
                if current_start is None:
                    current_start = dt
                prev_dt = dt
            else:
                if current_start is not None:
                    duration_min = (prev_dt - current_start).total_seconds() / 60
                    periods.append({
                        "start": current_start.strftime("%Y-%m-%d %H:%M"),
                        "end": prev_dt.strftime("%Y-%m-%d %H:%M"),
                        "duration_min": round(duration_min, 1),
                    })
                    current_start = None
                    prev_dt = None

        if current_start is not None:
            duration_min = (prev_dt - current_start).total_seconds() / 60
            periods.append({
                "start": current_start.strftime("%Y-%m-%d %H:%M"),
                "end": prev_dt.strftime("%Y-%m-%d %H:%M"),
                "duration_min": round(duration_min, 1),
            })

        total_duration = sum(p["duration_min"] for p in periods)
        max_duration = max((p["duration_min"] for p in periods), default=0)

        return {
            "periods": periods,
            "total_duration_min": round(total_duration, 1),
            "max_duration_min": round(max_duration, 1),
            "threshold": threshold,
            "operator": operator,
            "exceed_count": len(periods),
        }

    def analyze_comparison(
        self,
        multi_data: Dict[str, List[Dict]],
        device_info: Dict[str, Dict],
    ) -> Dict:
        """多序列对比分析"""
        comparison = {}
        all_datetimes = set()

        for tagname, data in multi_data.items():
            if not data:
                continue
            values = [p["value"] for p in data]
            info = device_info.get(tagname, {})

            stats = {
                "tagname": tagname,
                "label": info.get("cn_desc", tagname),
                "room_name": info.get("room_name", ""),
                "measure_label": info.get("measure_label", ""),
                "unit": info.get("unit", ""),
                "count": len(values),
                "current": values[-1] if values else None,
                "max": round(max(values), 2) if values else None,
                "min": round(min(values), 2) if values else None,
                "avg": round(sum(values) / len(values), 2) if values else None,
                "std_dev": round(float(np.std(values, ddof=1)), 2) if len(values) > 1 else 0,
            }
            comparison[tagname] = stats

            for p in data:
                all_datetimes.add(p["datetime"])

        sorted_dts = sorted(all_datetimes)
        chart_series = []
        for tagname, data in multi_data.items():
            if not data:
                continue
            dt_val_map = {p["datetime"]: p["value"] for p in data}
            info = device_info.get(tagname, {})
            series_data = [dt_val_map.get(dt) for dt in sorted_dts]
            chart_series.append({
                "name": info.get("cn_desc", tagname),
                "data": series_data,
            })

        return {
            "devices": comparison,
            "chart_categories": sorted_dts,
            "chart_series": chart_series,
        }

    def analyze_trend(self, data: List[Dict]) -> Dict:
        """趋势分析：变化率、阶跃检测、异常点检测

        使用ADTK LevelShiftAD检测阶跃变化，3σ检测离群点
        """
        if len(data) < 2:
            return {"trend": "insufficient_data", "rate_of_change": 0}

        values = [p["value"] for p in data]
        total_change = values[-1] - values[0]
        time_span_hours = len(data) / 60.0  # 假设1分钟间隔
        rate_per_hour = total_change / time_span_hours if time_span_hours > 0 else 0

        # 趋势判断：前后半段均值对比
        first_half_avg = sum(values[:len(values)//2]) / (len(values)//2) if len(values)//2 > 0 else 0
        second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2) if len(values) - len(values)//2 > 0 else 0

        if abs(second_half_avg - first_half_avg) < 0.1:
            trend = "stable"
        elif second_half_avg > first_half_avg:
            trend = "rising"
        else:
            trend = "declining"

        # 3σ异常点检测
        avg = sum(values) / len(values)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0
        outliers = []
        for p in data:
            if std > 0 and abs(p["value"] - avg) > 3 * std:
                outliers.append({"datetime": p["datetime"], "value": p["value"]})

        # ADTK阶跃检测
        level_shifts = []
        try:
            s = _to_series(data)
            s = validate_series(s)
            if len(s) >= 30:
                shift_detector = LevelShiftAD(c=3.0, side='both', window=10)
                shift_anomalies = shift_detector.detect(s, return_list=True)
                for event in shift_anomalies:
                    if isinstance(event, tuple):
                        level_shifts.append({
                            "start": event[0].strftime("%Y-%m-%d %H:%M"),
                            "end": event[1].strftime("%Y-%m-%d %H:%M"),
                        })
        except Exception as e:
            logger.debug(f"[TimeSeriesAnalyzer] LevelShiftAD检测跳过: {e}")

        # 小时聚合
        hourly = self._aggregate_hourly_pd(data)

        result = {
            "trend": trend,
            "total_change": round(total_change, 4),
            "rate_per_hour": round(rate_per_hour, 4),
            "first_half_avg": round(first_half_avg, 2),
            "second_half_avg": round(second_half_avg, 2),
            "outliers": outliers[:10],
            "level_shifts": level_shifts[:5],
            "hourly_avg": hourly,
        }

        return result

    def analyze_aggregation(
        self,
        data: List[Dict],
        interval: str = "hour",
    ) -> List[Dict]:
        """聚合统计：用Pandas resample按小时/天计算均值、最大、最小"""
        if not data:
            return []

        try:
            s = _to_series(data)
            rule = '1h' if interval == "hour" else '1D'

            agg_df = s.resample(rule).agg(['mean', 'max', 'min', 'count'])
            agg_df = agg_df.dropna(subset=[('mean',)])

            result = []
            for idx, row in agg_df.iterrows():
                key_field = "hour" if interval == "hour" else "date"
                fmt = "%Y-%m-%d %H:00" if interval == "hour" else "%Y-%m-%d"
                result.append({
                    key_field: idx.strftime(fmt),
                    "avg": round(float(row['mean']), 2),
                    "max": round(float(row['max']), 2),
                    "min": round(float(row['min']), 2),
                    "count": int(row['count']),
                })
            return result

        except Exception as e:
            logger.warning(f"[TimeSeriesAnalyzer] Pandas resample失败，降级: {e}")
            if interval == "day":
                return self._aggregate_daily_manual(data)
            return self._aggregate_hourly_manual(data)

    def analyze_anomaly(
        self,
        data: List[Dict],
        detectors: List[str] = None,
    ) -> Dict:
        """异常检测：组合多种ADTK检测器

        Args:
            data: [{datetime, value}, ...]
            detectors: 检测器列表，默认 ['persist', 'level_shift', 'volatility']
                - persist: 持续性异常（值持续偏离正常水平）
                - level_shift: 阶跃突变（值突然跳变）
                - volatility: 波动异常（方差突变）
                - seasonal: 季节性异常（偏离季节模式）
        """
        if not data or len(data) < 10:
            return {"anomalies": {}, "total_anomaly_points": 0}

        if detectors is None:
            detectors = ['persist', 'level_shift', 'volatility']

        try:
            s = _to_series(data)
            s = validate_series(s)
        except Exception as e:
            logger.warning(f"[TimeSeriesAnalyzer] validate_series失败: {e}")
            return {"anomalies": {}, "total_anomaly_points": 0}

        anomalies = {}
        all_anomaly_times = set()

        # 持续性异常检测
        if 'persist' in detectors:
            try:
                persist_ad = PersistAD(c=3.0, side='both')
                persist_result = persist_ad.detect(s, return_list=True)
                persist_events = self._format_events(persist_result, s)
                if persist_events:
                    anomalies["persist"] = persist_events
                    for ev in persist_events:
                        all_anomaly_times.add(ev.get("start", ""))
            except Exception as e:
                logger.debug(f"[TimeSeriesAnalyzer] PersistAD跳过: {e}")

        # 阶跃突变检测
        if 'level_shift' in detectors:
            try:
                window = min(10, max(3, len(s) // 20))
                shift_ad = LevelShiftAD(c=3.0, side='both', window=window)
                shift_result = shift_ad.detect(s, return_list=True)
                shift_events = self._format_events(shift_result, s)
                if shift_events:
                    anomalies["level_shift"] = shift_events
                    for ev in shift_events:
                        all_anomaly_times.add(ev.get("start", ""))
            except Exception as e:
                logger.debug(f"[TimeSeriesAnalyzer] LevelShiftAD跳过: {e}")

        # 波动异常检测
        if 'volatility' in detectors:
            try:
                window = min(10, max(3, len(s) // 20))
                vol_ad = VolatilityShiftAD(c=3.0, side='both', window=window)
                vol_result = vol_ad.detect(s, return_list=True)
                vol_events = self._format_events(vol_result, s)
                if vol_events:
                    anomalies["volatility"] = vol_events
                    for ev in vol_events:
                        all_anomaly_times.add(ev.get("start", ""))
            except Exception as e:
                logger.debug(f"[TimeSeriesAnalyzer] VolatilityShiftAD跳过: {e}")

        # 季节性异常检测
        if 'seasonal' in detectors and len(s) >= 48:
            try:
                seasonal_ad = SeasonalAD(c=3.0, side='both')
                seasonal_result = seasonal_ad.detect(s, return_list=True)
                seasonal_events = self._format_events(seasonal_result, s)
                if seasonal_events:
                    anomalies["seasonal"] = seasonal_events
                    for ev in seasonal_events:
                        all_anomaly_times.add(ev.get("start", ""))
            except Exception as e:
                logger.debug(f"[TimeSeriesAnalyzer] SeasonalAD跳过: {e}")

        return {
            "anomalies": anomalies,
            "total_anomaly_events": sum(len(v) for v in anomalies.values()),
            "detectors_used": list(anomalies.keys()),
        }

    # ==================== 查询解析 ====================

    def parse_scada_query(self, query: str) -> Dict:
        """解析SCADA相关查询，提取设备、时间、分析类型、阈值等参数"""
        result = {
            "devices": [],
            "tagnames": [],
            "device_info": {},
            "time_range": {},
            "analysis_type": self.ANALYSIS_RAW,
            "threshold": None,
            "threshold_operator": ">",
            "resolution": 60000,
        }

        registry = self._get_registry()

        # 1. 提取时间范围
        result["time_range"] = self._extract_time_range(query)

        # 2. 提取阈值
        threshold_patterns = [
            r'超过\s*(\d+\.?\d*)\s*[℃°摄氏度%]?',
            r'大于\s*(\d+\.?\d*)\s*[℃°摄氏度%]?',
            r'高于\s*(\d+\.?\d*)\s*[℃°摄氏度%]?',
            r'低于\s*(\d+\.?\d*)\s*[℃°摄氏度%]?',
            r'小于\s*(\d+\.?\d*)\s*[℃°摄氏度%]?',
            r'(\d+\.?\d*)\s*[℃°].*以上',
            r'(\d+\.?\d*)\s*[℃°].*以下',
        ]
        for pattern in threshold_patterns:
            m = re.search(pattern, query)
            if m:
                result["threshold"] = float(m.group(1))
                if '低于' in query or '小于' in query or '以下' in query:
                    result["threshold_operator"] = "<"
                else:
                    result["threshold_operator"] = ">"
                result["analysis_type"] = self.ANALYSIS_THRESHOLD
                break

        # 3. 异常检测意图
        if result["analysis_type"] == self.ANALYSIS_RAW:
            anomaly_keywords = ['异常', '突变', '阶跃', '波动异常', '不正常']
            for kw in anomaly_keywords:
                if kw in query:
                    result["analysis_type"] = self.ANALYSIS_ANOMALY
                    break

        # 4. 对比意图
        if result["analysis_type"] == self.ANALYSIS_RAW:
            comparison_keywords = ['对比', '比较', 'vs', '和.*比', '与.*比', '之间']
            for kw in comparison_keywords:
                if re.search(kw, query):
                    result["analysis_type"] = self.ANALYSIS_COMPARISON
                    break

        # 5. 趋势意图
        if result["analysis_type"] == self.ANALYSIS_RAW:
            trend_keywords = ['趋势', '变化', '走势', '波动']
            for kw in trend_keywords:
                if kw in query:
                    result["analysis_type"] = self.ANALYSIS_TREND
                    break

        # 6. 聚合意图
        if result["analysis_type"] == self.ANALYSIS_RAW:
            if re.search(r'平均|均值|每天|每小时|日均|周均|月均|汇总|统计', query):
                result["analysis_type"] = self.ANALYSIS_AGGREGATION

        # 7. 提取设备
        room_codes = re.findall(r'[A-Za-z]\d[A-Za-z]\d{2,3}', query)
        room_names = re.findall(
            r'(纯化间|培养间|配制间|缓冲间|清洗间|接种间|收获间|灭菌间|储藏间|'
            r'洁净区|洁净室|灌装间|冻干间|包装间|称量间|检验间|'
            r'空调机房|冷库|仓库|走廊|更衣间|气闸间)', query)

        measure_type_filter = None
        measure_keywords = {
            'TT': ['温度', 'temp'],
            'HT': ['湿度', 'humidity'],
            'PD': ['压差', 'pressure_diff'],
            'PDT': ['压差'],
            'PT': ['压力', 'pressure'],
            'FT': ['流量', 'flow'],
            'LT': ['液位', 'level'],
        }
        for mtype, keywords in measure_keywords.items():
            for kw in keywords:
                if kw in query.lower():
                    measure_type_filter = mtype
                    break
            if measure_type_filter:
                break

        # 优先用房间代码精确搜索；仅在没有房间代码时才用房间名称搜索
        if room_codes:
            search_queries = list(room_codes)
        else:
            search_queries = list(room_names)
        if not search_queries:
            search_queries = [query]

        for sq in search_queries:
            matches = registry.search(sq, top_k=10)
            for m in matches:
                tagname = m['tagname']
                measure_type = m.get('measure_type', 'PV')
                if measure_type_filter and measure_type != measure_type_filter:
                    if not (measure_type_filter == 'PD' and measure_type == 'PDT'):
                        continue
                # 如果有精确房间代码，过滤掉不属于这些房间的设备
                if room_codes:
                    device_room = m.get('room_code', '')
                    if device_room not in room_codes:
                        continue
                if tagname not in result["tagnames"]:
                    result["tagnames"].append(tagname)
                    measure_info = registry.get_measure_type_info(measure_type)
                    result["device_info"][tagname] = {
                        "tagname": tagname,
                        "cn_desc": m.get('cn_desc', tagname),
                        "room_code": m.get('room_code', ''),
                        "room_name": m.get('room_name', ''),
                        "measure_type": measure_type,
                        "measure_label": measure_info.get('label', ''),
                        "unit": measure_info.get('unit', ''),
                        "color": measure_info.get('color', '#4fc3f7'),
                    }

        # 8. 根据分析类型调整分辨率
        time_range = result["time_range"]
        start = time_range.get("start_date")
        end = time_range.get("end_date")
        if start and end:
            try:
                s = _parse_dt(start)
                e = _parse_dt(end)
                hours = (e - s).total_seconds() / 3600
                if hours > 168:
                    result["resolution"] = 900000
                elif hours > 72:
                    result["resolution"] = 300000
            except ValueError:
                pass

        return result

    def _extract_time_range(self, query: str) -> Dict:
        """从查询中提取时间范围"""
        now = datetime.now()
        result = {
            "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00"),
            "end_date": now.strftime("%Y-%m-%d %H:%M:%S"),
        }

        q = query
        if "昨天" in q or "昨日" in q:
            yesterday = now - timedelta(days=1)
            result["start_date"] = yesterday.strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = yesterday.strftime("%Y-%m-%d 23:59:59")
        elif "前天" in q:
            day = now - timedelta(days=2)
            result["start_date"] = day.strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = day.strftime("%Y-%m-%d 23:59:59")
        elif "本周" in q or "这周" in q:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            result["start_date"] = this_week_start.strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
        elif "上周" in q:
            weekday = now.weekday()
            this_week_start = now - timedelta(days=weekday)
            result["start_date"] = (this_week_start - timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = (this_week_start - timedelta(days=1)).strftime("%Y-%m-%d 23:59:59")
        elif "本月" in q:
            result["start_date"] = now.replace(day=1).strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
        elif "上月" in q:
            first_of_this_month = now.replace(day=1)
            last_month_end = first_of_this_month - timedelta(days=1)
            result["start_date"] = last_month_end.replace(day=1).strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = last_month_end.strftime("%Y-%m-%d 23:59:59")
        elif "今天" in q or "今日" in q:
            result["start_date"] = now.strftime("%Y-%m-%d 00:00:00")
            result["end_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            m = re.search(r'最近(\d+)天', q)
            if m:
                days = int(m.group(1))
                result["start_date"] = (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
                result["end_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
            m = re.search(r'(\d+)天前', q)
            if m:
                days = int(m.group(1))
                day = now - timedelta(days=days)
                result["start_date"] = day.strftime("%Y-%m-%d 00:00:00")
                result["end_date"] = day.strftime("%Y-%m-%d 23:59:59")

        return result

    # ==================== 图表配置 ====================

    def build_chart_config(
        self,
        multi_data: Dict[str, List[Dict]],
        device_info: Dict[str, Dict],
        analysis_type: str = "raw",
        threshold_info: Dict = None,
        aggregation_data: Dict[str, List[Dict]] = None,
        anomaly_data: Dict = None,
    ) -> Dict:
        """构建ECharts图表配置"""
        colors = ["#4fc3f7", "#66bb6a", "#ffb74d", "#ef5350", "#ab47bc", "#26c6da"]

        # 聚合统计模式
        if analysis_type == "aggregation" and aggregation_data:
            return self._build_agg_chart_config(aggregation_data, device_info, colors)

        # 异常检测模式
        if analysis_type == "anomaly" and anomaly_data:
            return self._build_anomaly_chart_config(multi_data, device_info, anomaly_data, colors)

        # 原始时序模式
        all_dts = set()
        for tagname, data in multi_data.items():
            for p in data:
                all_dts.add(p["datetime"])
        sorted_dts = sorted(all_dts)

        series = []
        for i, (tagname, data) in enumerate(multi_data.items()):
            if not data:
                continue
            dt_val_map = {p["datetime"]: p["value"] for p in data}
            info = device_info.get(tagname, {})
            color = info.get("color", colors[i % len(colors)])
            series_data = [dt_val_map.get(dt) for dt in sorted_dts]

            s = {
                "name": info.get("cn_desc", tagname),
                "type": "line",
                "data": series_data,
                "smooth": True,
                "symbol": "none",
                "lineStyle": {"width": 2, "color": color},
                "itemStyle": {"color": color},
            }

            if analysis_type != "comparison":
                s["areaStyle"] = {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": color + "40"},
                            {"offset": 1, "color": color + "05"},
                        ]
                    }
                }

            series.append(s)

        # 阈值标线
        if threshold_info and threshold_info.get("threshold") is not None:
            threshold_val = threshold_info["threshold"]
            markLine = {
                "silent": True,
                "lineStyle": {"color": "#ef5350", "type": "dashed", "width": 2},
                "label": {"formatter": f"阈值: {threshold_val}", "position": "insideEndTop"},
                "data": [{"yAxis": threshold_val}],
            }
            if series:
                series[0]["markLine"] = markLine

        # 趋势分析添加平均标线
        if analysis_type == "trend" and series:
            for s in series:
                s["markLine"] = {
                    "silent": True,
                    "data": [{"type": "average", "name": "平均"}],
                    "lineStyle": {"color": "#ffb74d", "type": "dashed"},
                }

        # 时间格式化
        categories = []
        time_span_days = 1
        if sorted_dts:
            try:
                first_dt = _parse_dt(sorted_dts[0])
                last_dt = _parse_dt(sorted_dts[-1])
                time_span_days = max(1, (last_dt - first_dt).days + 1)
            except ValueError:
                pass

        for dt_str in sorted_dts:
            try:
                dt = _parse_dt(dt_str)
                if time_span_days > 1:
                    categories.append(dt.strftime("%m/%d %H:%M"))
                else:
                    categories.append(dt.strftime("%H:%M"))
            except ValueError:
                categories.append(dt_str)

        label_interval = max(1, len(categories) // 10)

        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [s["name"] for s in series], "top": 5},
            "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"rotate": 0, "interval": label_interval, "fontSize": 11, "color": "#666"},
                "axisTick": {"alignWithLabel": True},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11}},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 5},
            ],
            "series": series,
        }

    @staticmethod
    def _build_agg_chart_config(aggregation_data, device_info, colors):
        """构建聚合统计图表配置（柱状图+折线图）"""
        series = []
        all_keys = set()

        for tagname, agg_list in aggregation_data.items():
            if not agg_list:
                continue
            for item in agg_list:
                key = item.get("hour") or item.get("date") or ""
                all_keys.add(key)

        sorted_keys = sorted(all_keys)
        categories = []
        for k in sorted_keys:
            try:
                dt = _parse_dt_hour(k)
                categories.append(dt.strftime("%m/%d %H:00"))
            except ValueError:
                try:
                    dt = _parse_dt_date(k)
                    categories.append(dt.strftime("%m/%d"))
                except ValueError:
                    categories.append(k[5:] if len(k) > 5 else k)

        for i, (tagname, agg_list) in enumerate(aggregation_data.items()):
            if not agg_list:
                continue
            info = device_info.get(tagname, {})
            color = info.get("color", colors[i % len(colors)])
            key_map = {}
            for item in agg_list:
                key = item.get("hour") or item.get("date") or ""
                key_map[key] = item

            avg_data = [round(key_map.get(k, {}).get("avg", 0), 2) for k in sorted_keys]
            series.append({
                "name": f"{info.get('cn_desc', tagname)} 均值",
                "type": "line",
                "data": avg_data,
                "smooth": True,
                "lineStyle": {"width": 2, "color": color},
                "itemStyle": {"color": color},
            })
            max_data = [round(key_map.get(k, {}).get("max", 0), 2) for k in sorted_keys]
            series.append({
                "name": f"{info.get('cn_desc', tagname)} 最高",
                "type": "bar",
                "data": max_data,
                "barMaxWidth": 20,
                "itemStyle": {"color": color + "60"},
            })

        label_interval = max(1, len(categories) // 12)

        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [s["name"] for s in series], "top": 5},
            "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"interval": label_interval, "fontSize": 11, "color": "#666"},
                "axisTick": {"alignWithLabel": True},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11}},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 5},
            ],
            "series": series,
        }

    @staticmethod
    def _build_anomaly_chart_config(multi_data, device_info, anomaly_data, colors):
        """构建异常检测图表配置（原始数据 + 异常标记）"""
        all_dts = set()
        for tagname, data in multi_data.items():
            for p in data:
                all_dts.add(p["datetime"])
        sorted_dts = sorted(all_dts)

        series = []
        for i, (tagname, data) in enumerate(multi_data.items()):
            if not data:
                continue
            dt_val_map = {p["datetime"]: p["value"] for p in data}
            info = device_info.get(tagname, {})
            color = info.get("color", colors[i % len(colors)])
            series_data = [dt_val_map.get(dt) for dt in sorted_dts]

            series.append({
                "name": info.get("cn_desc", tagname),
                "type": "line",
                "data": series_data,
                "smooth": True,
                "symbol": "none",
                "lineStyle": {"width": 2, "color": color},
                "itemStyle": {"color": color},
                "areaStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": color + "40"},
                            {"offset": 1, "color": color + "05"},
                        ]
                    }
                },
            })

        # 添加异常标记系列
        anomaly_types = anomaly_data.get("anomalies", {})
        type_colors = {
            "persist": "#ef5350",
            "level_shift": "#ff9800",
            "volatility": "#9c27b0",
            "seasonal": "#2196f3",
        }
        for atype, events in anomaly_types.items():
            mark_data = []
            for ev in events:
                start_str = ev.get("start", "")
                # 在categories中找到对应索引
                for j, dt_str in enumerate(sorted_dts):
                    if dt_str.startswith(start_str[:10]):
                        mark_data.append({
                            "coord": [j, None],
                            "symbol": "circle",
                            "symbolSize": 8,
                            "itemStyle": {"color": type_colors.get(atype, "#ef5350")},
                        })
                        break

            if mark_data and series:
                series[0].setdefault("markPoint", {
                    "symbol": "circle",
                    "symbolSize": 8,
                    "data": [],
                    "label": {"show": False},
                })
                series[0]["markPoint"]["data"].extend(mark_data)

        categories = []
        for dt_str in sorted_dts:
            try:
                dt = _parse_dt(dt_str)
                categories.append(dt.strftime("%H:%M"))
            except ValueError:
                categories.append(dt_str)

        label_interval = max(1, len(categories) // 10)

        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [s["name"] for s in series], "top": 5},
            "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"interval": label_interval, "fontSize": 11, "color": "#666"},
                "axisTick": {"alignWithLabel": True},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11}},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 5},
            ],
            "series": series,
        }

    # ==================== 辅助方法 ====================

    @staticmethod
    def _format_events(events, series: pd.Series) -> List[Dict]:
        """将ADTK事件列表转为统一格式"""
        result = []
        for event in events:
            if isinstance(event, tuple):
                start, end = event
                mask = (series.index >= start) & (series.index <= end)
                period_values = series[mask]
                avg_val = round(float(period_values.mean()), 2) if len(period_values) > 0 else None
                max_val = round(float(period_values.max()), 2) if len(period_values) > 0 else None
                result.append({
                    "start": start.strftime("%Y-%m-%d %H:%M"),
                    "end": end.strftime("%Y-%m-%d %H:%M"),
                    "duration_min": round((end - start).total_seconds() / 60, 1),
                    "avg_value": avg_val,
                    "max_value": max_val,
                })
            elif isinstance(event, pd.Timestamp):
                val = series.get(event, None)
                result.append({
                    "start": event.strftime("%Y-%m-%d %H:%M"),
                    "end": event.strftime("%Y-%m-%d %H:%M"),
                    "duration_min": 1,
                    "avg_value": round(float(val), 2) if pd.notna(val) else None,
                    "max_value": round(float(val), 2) if pd.notna(val) else None,
                })
        return result

    @staticmethod
    def _aggregate_hourly_pd(data: List[Dict]) -> List[Dict]:
        """用Pandas resample按小时聚合"""
        if not data:
            return []
        try:
            s = _to_series(data)
            agg_df = s.resample('1h').agg(['mean', 'max', 'min', 'count'])
            agg_df = agg_df.dropna(subset=[('mean',)])
            result = []
            for idx, row in agg_df.iterrows():
                result.append({
                    "hour": idx.strftime("%Y-%m-%d %H:00"),
                    "avg": round(float(row['mean']), 2),
                    "max": round(float(row['max']), 2),
                    "min": round(float(row['min']), 2),
                    "count": int(row['count']),
                })
            return result
        except Exception:
            return TimeSeriesAnalyzer._aggregate_hourly_manual(data)

    @staticmethod
    def _aggregate_hourly_manual(data: List[Dict]) -> List[Dict]:
        """手动按小时聚合（降级方案）"""
        hourly = {}
        for p in data:
            try:
                dt = _parse_dt(p["datetime"])
                hour_key = dt.strftime("%Y-%m-%d %H:00")
            except ValueError:
                continue
            if hour_key not in hourly:
                hourly[hour_key] = []
            hourly[hour_key].append(p["value"])

        result = []
        for hour_key in sorted(hourly.keys()):
            values = hourly[hour_key]
            result.append({
                "hour": hour_key,
                "avg": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "min": round(min(values), 2),
                "count": len(values),
            })
        return result

    @staticmethod
    def _aggregate_daily_manual(data: List[Dict]) -> List[Dict]:
        """手动按天聚合（降级方案）"""
        daily = {}
        for p in data:
            try:
                dt = _parse_dt(p["datetime"])
                day_key = dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
            if day_key not in daily:
                daily[day_key] = []
            daily[day_key].append(p["value"])

        result = []
        for day_key in sorted(daily.keys()):
            values = daily[day_key]
            result.append({
                "date": day_key,
                "avg": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "min": round(min(values), 2),
                "count": len(values),
            })
        return result


# 向后兼容别名
SCADAAnalyzer = TimeSeriesAnalyzer
