"""
FMCS设备数据点注册表

从Excel加载所有设备数据点，支持自然语言匹配到具体TagName
支持的数据类型：温度、湿度、压差、压力、流量、液位、转速、频率、阀门、状态等
"""
import os
import re
from typing import List, Optional, Dict, Tuple
from utils.logger import logger


# 数据类型映射：后缀 → 中文类型 + 单位 + 图表颜色
MEASUREMENT_TYPES = {
    'TT': {'label': '温度', 'unit': '°C', 'color': '#4fc3f7', 'area_color': 'rgba(79,195,247,0.3)'},
    'HT': {'label': '湿度', 'unit': '%RH', 'color': '#66bb6a', 'area_color': 'rgba(102,187,106,0.3)'},
    'PDT': {'label': '压差', 'unit': 'Pa', 'color': '#ffb74d', 'area_color': 'rgba(255,183,77,0.3)'},
    'PD': {'label': '压差', 'unit': 'Pa', 'color': '#ffb74d', 'area_color': 'rgba(255,183,77,0.3)'},
    'PT': {'label': '压力', 'unit': 'Pa', 'color': '#ef5350', 'area_color': 'rgba(239,83,80,0.3)'},
    'FT': {'label': '流量', 'unit': 'm³/h', 'color': '#ab47bc', 'area_color': 'rgba(171,71,188,0.3)'},
    'LT': {'label': '液位', 'unit': 'm', 'color': '#26c6da', 'area_color': 'rgba(38,198,218,0.3)'},
    'ST': {'label': '转速', 'unit': 'rpm', 'color': '#ff7043', 'area_color': 'rgba(255,112,67,0.3)'},
    'FRQ': {'label': '频率', 'unit': 'Hz', 'color': '#7e57c2', 'area_color': 'rgba(126,87,194,0.3)'},
    'PV': {'label': '过程值', 'unit': '', 'color': '#42a5f5', 'area_color': 'rgba(66,165,245,0.3)'},
}

# 中文关键词 → 数据类型映射
KEYWORD_TO_TYPE = {
    '温度': 'TT', 'temp': 'TT', '气温': 'TT', '室温': 'TT',
    '湿度': 'HT', 'humidity': 'HT', '相对湿度': 'HT',
    '压差': 'PD', '压力差': 'PD', '差压': 'PD',
    '压力': 'PT', 'pressure': 'PT', '气压': 'PT', '风压': 'PT',
    '流量': 'FT', 'flow': 'FT', '风量': 'FT', '水流量': 'FT',
    '液位': 'LT', 'level': 'LT', '水位': 'LT', '油位': 'LT',
    '转速': 'ST', 'speed': 'ST', 'RPM': 'ST', 'rpm': 'ST',
    '频率': 'FRQ', 'frequency': 'FRQ', 'Hz': 'FRQ', '赫兹': 'FRQ',
}


class FMCSDeviceRegistry:
    """FMCS设备数据点注册表"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.devices: List[Dict] = []
        self._tagname_index: Dict[str, Dict] = {}  # tagname → device info
        self._room_index: Dict[str, List[Dict]] = {}  # room_code → [devices]
        self._cn_keyword_index: Dict[str, List[Dict]] = {}  # 中文关键词 → [devices]
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_from_excel()

    def load_from_excel(self, excel_path: str = None):
        """从Excel加载数据点"""
        if excel_path is None:
            excel_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'FMCS数据点.xlsx'
            )

        if not os.path.exists(excel_path):
            logger.warning(f"[FMCS] Excel文件不存在: {excel_path}")
            return

        try:
            import openpyxl
            wb = openpyxl.load_workbook(excel_path, read_only=True)
            ws = wb.active

            self.devices = []
            self._tagname_index = {}
            self._room_index = {}
            self._cn_keyword_index = {}

            for row in ws.iter_rows(min_row=2, values_only=True):
                tagname = str(row[0]).strip() if row[0] else ''
                desc = str(row[1]).strip() if row[1] else ''

                if not tagname or tagname == 'None':
                    continue

                # 解析描述: "A1S115_TT.PV || A1S115纯化间 II 温度监测 / A1S115 Temperature in Purification II"
                cn_desc = ''
                en_desc = ''
                if '||' in desc:
                    parts = desc.split('||', 1)
                    cn_part = parts[1].strip()
                    if '/' in cn_part:
                        cn_desc = cn_part.split('/')[0].strip()
                        en_desc = cn_part.split('/', 1)[1].strip()
                    else:
                        cn_desc = cn_part

                # 解析测量类型
                measure_type = self._parse_measure_type(tagname, cn_desc)

                # 解析房间/区域代码
                room_code = self._parse_room_code(tagname, cn_desc)

                # 解析房间中文名
                room_name = self._parse_room_name(cn_desc)

                device = {
                    'tagname': tagname,
                    'desc': desc,
                    'cn_desc': cn_desc,
                    'en_desc': en_desc,
                    'measure_type': measure_type,
                    'room_code': room_code,
                    'room_name': room_name,
                }

                self.devices.append(device)
                self._tagname_index[tagname] = device

                # 建立房间索引
                if room_code:
                    if room_code not in self._room_index:
                        self._room_index[room_code] = []
                    self._room_index[room_code].append(device)

                # 建立中文关键词索引
                for kw in self._extract_cn_keywords(cn_desc, room_name, tagname):
                    if kw not in self._cn_keyword_index:
                        self._cn_keyword_index[kw] = []
                    self._cn_keyword_index[kw].append(device)

            wb.close()
            self._loaded = True
            logger.info(f"[FMCS] 加载完成: {len(self.devices)} 个数据点, "
                        f"{len(self._room_index)} 个房间, "
                        f"{len(self._cn_keyword_index)} 个关键词")

        except Exception as e:
            logger.error(f"[FMCS] 加载Excel失败: {e}")

    def _parse_measure_type(self, tagname: str, cn_desc: str) -> str:
        """从TagName和中文描述解析测量类型"""
        # 从TagName后缀推断
        if '.' in tagname:
            prefix = tagname.split('.')[0]  # e.g. A1S115_TT
            # 提取最后一段下划线后的类型标识
            parts = prefix.split('_')
            if len(parts) >= 2:
                type_code = parts[-1]  # e.g. TT, HT, PDT, PD1, PT
                # 标准化
                if type_code.startswith('PD'):
                    return 'PD'
                if type_code in MEASUREMENT_TYPES:
                    return type_code

        # 从中文描述推断
        for kw, type_code in KEYWORD_TO_TYPE.items():
            if kw in cn_desc:
                return type_code

        return 'PV'  # 默认

    def _parse_room_code(self, tagname: str, cn_desc: str) -> str:
        """解析房间/区域代码"""
        # 从TagName前缀提取: A1S115_TT.PV → A1S115
        if '.' in tagname:
            prefix = tagname.split('.')[0]
            parts = prefix.split('_')
            if len(parts) >= 2:
                return parts[0]  # e.g. A1S115
        # 从中文描述提取
        m = re.match(r'([A-Z]\d+[A-Z]\d+)', cn_desc)
        if m:
            return m.group(1)
        return ''

    def _parse_room_name(self, cn_desc: str) -> str:
        """从中文描述提取房间名称"""
        # 去掉开头的房间代码，提取纯房间名
        # "A1S115纯化间 II 温度监测" → "纯化间 II"
        m = re.match(r'[A-Z]\d+[A-Z]\d+\s*(.+?)(?:温度|湿度|压差|压力|流量|液位|转速|频率|监测|控制|状态|开关|运行|累计)', cn_desc)
        if m:
            return m.group(1).strip()
        return ''

    def _extract_cn_keywords(self, cn_desc: str, room_name: str, tagname: str) -> List[str]:
        """提取中文关键词用于索引"""
        keywords = set()

        # 房间名
        if room_name:
            keywords.add(room_name)
            # 纯化间 → 纯化
            for i in range(len(room_name)):
                for j in range(i + 1, min(i + 5, len(room_name) + 1)):
                    sub = room_name[i:j]
                    if len(sub) >= 2:
                        keywords.add(sub)

        # 描述中的关键词
        for kw in ['温度', '湿度', '压差', '压力', '流量', '液位', '转速', '频率',
                    '纯化', '培养', '接种', '收获', '配制', '缓冲', '清洗', '更衣',
                    '走廊', '缓冲间', '物流', '退出', '洁净', '细胞', '培养基',
                    '冷库', '冻存', '仓库', '灭菌', '灌装', '冻干', '包装',
                    '水系统', '纯蒸汽', '空压', '冷水', '热水', '空调', '新风',
                    '排风', '送风', '回风', '排风', '冷却', '加热', '加湿',
                    '除湿', '过滤', '消毒']:
            if kw in cn_desc:
                keywords.add(kw)

        # TagName中的房间代码
        if '.' in tagname:
            prefix = tagname.split('.')[0]
            parts = prefix.split('_')
            if len(parts) >= 2:
                keywords.add(parts[0])  # e.g. A1S115
                # 也加完整前缀
                keywords.add(prefix)  # e.g. A1S115_TT

        return list(keywords)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        自然语言搜索数据点

        支持的查询模式:
        - "A1S115温度" → 匹配房间+类型
        - "纯化间温度" → 匹配房间名+类型
        - "细胞培养间湿度" → 匹配房间名+类型
        - "A1S115所有数据" → 匹配房间所有数据点
        - "CC2C4_001.Temp" → 精确匹配TagName
        """
        self._ensure_loaded()

        q = query.strip()
        results = []
        scored: Dict[str, float] = {}  # tagname → score

        # 1. 精确TagName匹配
        if q in self._tagname_index:
            device = self._tagname_index[q]
            scored[q] = 100.0

        # 2. TagName前缀匹配 (e.g. "A1S115" 匹配 "A1S115_TT.PV")
        tag_prefix = q.replace('.', '_').split('_')[0].upper()
        for device in self.devices:
            if device['tagname'].upper().startswith(tag_prefix + '_') or \
               device['tagname'].upper().startswith(tag_prefix + '.'):
                tagname = device['tagname']
                if tagname not in scored:
                    scored[tagname] = 50.0
                else:
                    scored[tagname] = max(scored[tagname], 50.0)

        # 3. 解析查询中的测量类型
        target_type = None
        for kw, type_code in KEYWORD_TO_TYPE.items():
            if kw in q:
                target_type = type_code
                break

        # 4. 中文关键词匹配
        for kw, devices in self._cn_keyword_index.items():
            if kw in q:
                for device in devices:
                    tagname = device['tagname']
                    # 基础匹配分
                    score = 10.0
                    # 类型匹配加分
                    if target_type and device['measure_type'] == target_type:
                        score += 30.0
                    # 关键词长度加分（越长越精确）
                    score += len(kw) * 2
                    # 房间名完全匹配加分
                    if device['room_name'] and device['room_name'] in q:
                        score += 20.0

                    if tagname in scored:
                        scored[tagname] = max(scored[tagname], score)
                    else:
                        scored[tagname] = score

        # 5. 房间代码索引匹配
        for room_code, devices in self._room_index.items():
            if room_code in q:
                for device in devices:
                    tagname = device['tagname']
                    score = 15.0
                    if target_type and device['measure_type'] == target_type:
                        score += 30.0
                    if tagname in scored:
                        scored[tagname] = max(scored[tagname], score)
                    else:
                        scored[tagname] = score

        # 排序取top_k
        sorted_items = sorted(scored.items(), key=lambda x: -x[1])
        for tagname, score in sorted_items[:top_k]:
            device = self._tagname_index.get(tagname)
            if device:
                result = dict(device)
                result['match_score'] = score
                result['measure_label'] = MEASUREMENT_TYPES.get(
                    device['measure_type'], MEASUREMENT_TYPES['PV']
                )['label']
                result['unit'] = MEASUREMENT_TYPES.get(
                    device['measure_type'], MEASUREMENT_TYPES['PV']
                )['unit']
                results.append(result)

        return results

    def get_device(self, tagname: str) -> Optional[Dict]:
        """精确获取设备信息"""
        self._ensure_loaded()
        return self._tagname_index.get(tagname)

    def get_room_devices(self, room_code: str, measure_type: str = None) -> List[Dict]:
        """获取某房间/区域的所有数据点"""
        self._ensure_loaded()
        devices = self._room_index.get(room_code, [])
        if measure_type:
            devices = [d for d in devices if d['measure_type'] == measure_type]
        return devices

    def list_rooms(self) -> List[Dict]:
        """列出所有房间/区域及其数据点数量"""
        self._ensure_loaded()
        result = []
        for room_code, devices in sorted(self._room_index.items()):
            type_counts = {}
            for d in devices:
                t = d['measure_type']
                type_counts[t] = type_counts.get(t, 0) + 1
            room_name = devices[0]['room_name'] if devices else ''
            result.append({
                'room_code': room_code,
                'room_name': room_name,
                'device_count': len(devices),
                'types': type_counts,
            })
        return result

    def get_measure_type_info(self, measure_type: str) -> Dict:
        """获取测量类型信息"""
        return MEASUREMENT_TYPES.get(measure_type, MEASUREMENT_TYPES['PV'])


# 全局实例
_fmcs_registry = None


def get_fmcs_registry() -> FMCSDeviceRegistry:
    global _fmcs_registry
    if _fmcs_registry is None:
        _fmcs_registry = FMCSDeviceRegistry()
    return _fmcs_registry
