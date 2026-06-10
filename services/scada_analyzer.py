"""
SCADA智能分析服务 - 支持多设备查询、阈值分析、对比分析、趋势分析
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import logger


class SCADAAnalyzer:
    """SCADA数据智能分析引擎"""

    # 分析类型
    ANALYSIS_RAW = "raw"              # 原始时序数据
    ANALYSIS_THRESHOLD = "threshold"  # 阈值分析（超标持续时长等）
    ANALYSIS_COMPARISON = "comparison"  # 多设备对比
    ANALYSIS_TREND = "trend"          # 趋势分析（变化率、周期性）
    ANALYSIS_AGGREGATION = "aggregation"  # 聚合统计（小时/天均值）

    def __init__(self, historian_config=None):
        self._config = historian_config
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            from utils.fmcs_registry import FMCSDeviceRegistry
            self._registry = FMCSDeviceRegistry()
        return self._registry

    def fetch_timeseries(
        self,
        tagnames: List[str],
        start_date: str,
        end_date: str,
        resolution: int = 60000,
    ) -> Dict[str, List[Dict]]:
        """
        从Historian查询多个TagName的时序数据
        返回: {tagname: [{datetime, value}, ...]}
        """
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
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    data = []
                    for row in rows:
                        dt = row[0]
                        val = row[1]
                        if isinstance(dt, datetime):
                            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            dt_str = str(dt)
                        try:
                            val = round(float(val), 4)
                        except (ValueError, TypeError):
                            continue
                        data.append({"datetime": dt_str, "value": val})
                    result[tagname] = data
                    logger.info(f"[SCADAAnalyzer] {tagname}: 获取{len(data)}个数据点")
                except Exception as e:
                    logger.error(f"[SCADAAnalyzer] 查询{tagname}失败: {e}")
                    result[tagname] = []

            conn.close()
        except Exception as e:
            logger.error(f"[SCADAAnalyzer] Historian连接失败: {e}")
            return result

        return result

    def analyze_threshold(
        self,
        data: List[Dict],
        threshold: float,
        operator: str = ">",
    ) -> Dict:
        """
        阈值分析：找出数据超过/低于阈值的时段及持续时长
        operator: ">" 超过, "<" 低于, ">=" 大于等于, "<=" 小于等于
        """
        if not data:
            return {"periods": [], "total_duration_min": 0, "max_duration_min": 0}

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
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

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
                        "max_value": max(
                            p["value"] for p in data
                            if current_start <= datetime.strptime(p["datetime"], "%Y-%m-%d %H:%M:%S") <= prev_dt
                            and op_func(p["value"], threshold)
                        ) if duration_min > 0 else val,
                    })
                    current_start = None
                    prev_dt = None

        # 处理末尾仍在超标的时段
        if current_start is not None:
            duration_min = (prev_dt - current_start).total_seconds() / 60
            # 计算末尾时段的最大值
            tail_max = max(
                (p["value"] for p in data
                 if current_start <= datetime.strptime(p["datetime"], "%Y-%m-%d %H:%M:%S") <= prev_dt
                 and op_func(p["value"], threshold)),
                default=None
            )
            period_data = {
                "start": current_start.strftime("%Y-%m-%d %H:%M"),
                "end": prev_dt.strftime("%Y-%m-%d %H:%M"),
                "duration_min": round(duration_min, 1),
            }
            if tail_max is not None:
                period_data["max_value"] = round(tail_max, 2)
            periods.append(period_data)

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
        """
        多设备对比分析
        返回各设备的统计信息和对比图表数据
        """
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
                "std_dev": round(self._std_dev(values), 2) if len(values) > 1 else 0,
            }
            comparison[tagname] = stats

            for p in data:
                all_datetimes.add(p["datetime"])

        # 构建对齐的时序数据（用于图表）
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
        """
        趋势分析：变化率、周期性、异常点检测
        """
        if len(data) < 2:
            return {"trend": "insufficient_data", "rate_of_change": 0}

        values = [p["value"] for p in data]
        total_change = values[-1] - values[0]
        time_span_hours = len(data) / 60.0  # 假设1分钟间隔
        rate_per_hour = total_change / time_span_hours if time_span_hours > 0 else 0

        # 简单趋势判断
        first_half_avg = sum(values[:len(values)//2]) / (len(values)//2) if len(values)//2 > 0 else 0
        second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2) if len(values) - len(values)//2 > 0 else 0

        if abs(second_half_avg - first_half_avg) < 0.1:
            trend = "stable"
        elif second_half_avg > first_half_avg:
            trend = "rising"
        else:
            trend = "declining"

        # 异常点检测（3σ原则）
        avg = sum(values) / len(values)
        std = self._std_dev(values)
        outliers = []
        for p in data:
            if std > 0 and abs(p["value"] - avg) > 3 * std:
                outliers.append({"datetime": p["datetime"], "value": p["value"]})

        # 小时聚合
        hourly = self._aggregate_hourly(data)

        return {
            "trend": trend,
            "total_change": round(total_change, 4),
            "rate_per_hour": round(rate_per_hour, 4),
            "first_half_avg": round(first_half_avg, 2),
            "second_half_avg": round(second_half_avg, 2),
            "outliers": outliers[:10],
            "hourly_avg": hourly,
        }

    def analyze_aggregation(
        self,
        data: List[Dict],
        interval: str = "hour",
    ) -> List[Dict]:
        """
        聚合统计：按小时/天计算均值、最大、最小
        """
        if interval == "day":
            return self._aggregate_daily(data)
        return self._aggregate_hourly(data)

    def parse_scada_query(self, query: str) -> Dict:
        """
        解析SCADA相关查询，提取设备、时间、分析类型、阈值等参数
        """
        result = {
            "devices": [],          # 设备描述列表
            "tagnames": [],         # 匹配的TagName列表
            "device_info": {},      # tagname → 设备信息
            "time_range": {},       # start_date, end_date
            "analysis_type": self.ANALYSIS_RAW,
            "threshold": None,
            "threshold_operator": ">",
            "resolution": 60000,    # 默认1分钟
        }

        registry = self._get_registry()

        # 1. 提取时间范围
        result["time_range"] = self._extract_time_range(query)

        # 2. 提取阈值
        threshold_patterns = [
            r'超过\s*(\d+\.?\d*)\s*[℃°摄氏度]?',   # 超过22.4℃ / 超过22摄氏度
            r'大于\s*(\d+\.?\d*)\s*[℃°摄氏度]?',   # 大于22.4
            r'高于\s*(\d+\.?\d*)\s*[℃°摄氏度]?',   # 高于22.4
            r'低于\s*(\d+\.?\d*)\s*[℃°摄氏度]?',   # 低于22.4
            r'小于\s*(\d+\.?\d*)\s*[℃°摄氏度]?',   # 小于22.4
            r'(\d+\.?\d*)\s*[℃°摄氏度].*以上',      # 22.4℃以上
            r'(\d+\.?\d*)\s*[℃°摄氏度].*以下',      # 22.4℃以下
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

        # 3. 检测对比意图
        comparison_keywords = ['对比', '比较', 'vs', '和.*比', '与.*比', '之间']
        for kw in comparison_keywords:
            if re.search(kw, query):
                result["analysis_type"] = self.ANALYSIS_COMPARISON
                break

        # 4. 检测趋势意图
        trend_keywords = ['趋势', '变化', '走势', '波动']
        for kw in trend_keywords:
            if kw in query:
                result["analysis_type"] = self.ANALYSIS_TREND
                break

        # 5. 提取设备
        # 先尝试房间代码匹配
        room_codes = re.findall(r'[A-Za-z]\d[A-Za-z]\d{2,3}', query)
        # 再尝试中文房间名（扩展列表，覆盖更多房间）
        room_names = re.findall(
            r'(纯化间|培养间|配制间|缓冲间|清洗间|接种间|收获间|灭菌间|储藏间|'
            r'洁净区|洁净室|灌装间|冻干间|包装间|称量间|检验间|'
            r'空调机房|冷库|仓库|走廊|更衣间|气闸间)', query)

        # 提取用户提到的测量类型
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

        search_queries = list(room_codes) + list(room_names)

        # 如果没有明确的房间代码/名称，用整个查询搜索
        if not search_queries:
            search_queries = [query]

        for sq in search_queries:
            matches = registry.search(sq, top_k=10)
            for m in matches:
                tagname = m['tagname']
                measure_type = m.get('measure_type', 'PV')

                # 如果用户指定了测量类型，只保留匹配的
                if measure_type_filter and measure_type != measure_type_filter:
                    # PDT也匹配PD
                    if not (measure_type_filter == 'PD' and measure_type == 'PDT'):
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

        # 6. 根据分析类型调整分辨率
        time_range = result["time_range"]
        start = time_range.get("start_date")
        end = time_range.get("end_date")
        if start and end:
            try:
                s = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
                e = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
                hours = (e - s).total_seconds() / 3600
                if hours > 72:  # 超过3天，用5分钟分辨率
                    result["resolution"] = 300000
                elif hours > 168:  # 超过7天，用15分钟分辨率
                    result["resolution"] = 900000
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
            # 最近N天
            m = re.search(r'最近(\d+)天', q)
            if m:
                days = int(m.group(1))
                result["start_date"] = (now - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
                result["end_date"] = now.strftime("%Y-%m-%d %H:%M:%S")
            # N天前
            m = re.search(r'(\d+)天前', q)
            if m:
                days = int(m.group(1))
                day = now - timedelta(days=days)
                result["start_date"] = day.strftime("%Y-%m-%d 00:00:00")
                result["end_date"] = day.strftime("%Y-%m-%d 23:59:59")

        return result

    def build_chart_config(
        self,
        multi_data: Dict[str, List[Dict]],
        device_info: Dict[str, Dict],
        analysis_type: str = "raw",
        threshold_info: Dict = None,
    ) -> Dict:
        """构建ECharts图表配置"""
        # 收集所有时间点
        all_dts = set()
        for tagname, data in multi_data.items():
            for p in data:
                all_dts.add(p["datetime"])
        sorted_dts = sorted(all_dts)

        # 构建系列
        series = []
        colors = ["#4fc3f7", "#66bb6a", "#ffb74d", "#ef5350", "#ab47bc", "#26c6da"]
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

        # 阈值标线
        markLine = None
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

        # 时间格式化 - 根据时间跨度选择格式
        categories = []
        time_span_days = 1
        if sorted_dts:
            try:
                first_dt = datetime.strptime(sorted_dts[0], "%Y-%m-%d %H:%M:%S")
                last_dt = datetime.strptime(sorted_dts[-1], "%Y-%m-%d %H:%M:%S")
                time_span_days = max(1, (last_dt - first_dt).days + 1)
            except ValueError:
                pass

        for dt_str in sorted_dts:
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                if time_span_days > 1:
                    categories.append(dt.strftime("%m/%d %H:%M"))
                else:
                    categories.append(dt.strftime("%H:%M"))
            except ValueError:
                categories.append(dt_str)

        # 根据数据点数量决定x轴标签间隔
        label_interval = max(1, len(categories) // 10)

        config = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [s["name"] for s in series], "top": 5},
            "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "15%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {
                    "rotate": 0,
                    "interval": label_interval,
                    "fontSize": 11,
                    "color": "#666",
                },
                "axisTick": {"alignWithLabel": True},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11}},
            "dataZoom": [
                {"type": "inside", "start": 0, "end": 100},
                {"type": "slider", "start": 0, "end": 100, "height": 20, "bottom": 5},
            ],
            "series": series,
        }

        return config

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        if len(values) < 2:
            return 0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def _aggregate_hourly(data: List[Dict]) -> List[Dict]:
        """按小时聚合"""
        hourly = {}
        for p in data:
            try:
                dt = datetime.strptime(p["datetime"], "%Y-%m-%d %H:%M:%S")
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
    def _aggregate_daily(data: List[Dict]) -> List[Dict]:
        """按天聚合"""
        daily = {}
        for p in data:
            try:
                dt = datetime.strptime(p["datetime"], "%Y-%m-%d %H:%M:%S")
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
