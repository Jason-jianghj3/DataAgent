# 📊 检验报告智能解读与趋势口语播报

> 基于帆软报表数据，利用大语言模型自动生成夜班/日班检验结果的**口语化总结摘要**，并通过语音或文字推送至管理终端。

## ✨ 功能亮点

- **🤖 AI智能总结**：调用LLM将检验数据转换为200字左右的自然语言汇报
- **🔊 语音播报**：支持Edge TTS（免费）/Azure TTS/离线TTS多种引擎
- **📱 多渠道推送**：支持钉钉、飞书、企业微信、邮件等通知方式
- **🔌 一键集成**：帆软报表加个按钮即可调用，开发周期1周内
- **⚡ 降级保障**：LLM不可用时自动切换模板输出，保证可用性

## 📁 项目结构

```
帆软报表数据智能总结播报/
├── app.py                      # Flask Web API 服务入口
├── config.py                   # 配置文件（支持环境变量）
├── report_summary.py           # 主业务逻辑编排（流水线）
│
├── services/                   # 核心服务模块
│   ├── fine_report_client.py   # 帆软报表数据获取客户端
│   ├── llm_service.py          # LLM数据总结服务
│   ├── tts_service.py          # TTS语音合成服务
│   └── notifier.py             # 多渠道通知推送服务
│
├── utils/                      # 工具模块
│   └── logger.py               # 日志工具
│
├── audio_output/               # 生成的音频文件目录
├── logs/                       # 日志文件目录
├── .env.example                # 环境配置模板 ⭐ 复制为 .env 使用
├── requirements.txt            # Python依赖列表
├── start.bat                   # Windows一键启动脚本
└── README.md                   # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API Key等信息
```

**最少配置**：只需填写 `LLM_API_KEY` 即可运行演示模式。

### 3. 启动服务

**Windows：**
```bash
start.bat
```

**Linux/Mac：**
```bash
python app.py
```

### 4. 测试接口

服务启动后访问 `http://localhost:5000/api/health` 查看状态。

## 📡 API 接口说明

### 核心接口：生成完整摘要

```
POST /api/report/summary
Content-Type: application/json

{
    "shift": "夜班",              // 班次: 夜班/白班
    "include_tts": true,           // 是否生成语音 (默认true)
    "include_notify": false,       // 是否推送通知 (默认false)
    "product_name": "",            // 产品名称筛选(可选)
    "workshop": ""                 // 车间筛选(可选)
}
```

**响应示例：**
```json
{
    "success": true,
    "request_id": "20260424141900",
    "shift": "夜班",
    "summary_text": "夜班整体运行平稳。重点提示：纯化3号层析柱收率较前日下降1.5%...",
    "audio_url": "/audio/report_夜班_20260424141900.mp3",
    "data_fetch": {"success": true, "record_count": 10},
    "llm_summary": {"success": true, "token_usage": {...}},
    "tts_audio": {"success": true, "duration_seconds": 12.5},
    "notification": {}
}
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 配置状态 |
| GET/POST | `/api/report/text-only` | 仅生成文本摘要（不含语音） |
| POST | `/api/fr/callback` | **帆软回调专用接口** |
| GET | `/api/config` | 查看当前配置（脱敏） |
| GET | `/audio/<filename>` | 获取生成的音频文件 |
| GET | `/api/audio/list` | 列出已生成的音频 |

## 🔗 帆软报表集成方法

### 方式一：JS按钮调用（推荐）

在帆软报表的「按钮点击事件」中添加以下JavaScript：

```javascript
// 获取摘要文本
$.ajax({
    url: 'http://your-server:5000/api/report/summary',
    type: 'POST',
    data: JSON.stringify({ shift: '夜班' }),
    contentType: 'application/json',
    success: function(res) {
        if (res.success) {
            // 显示摘要文本
            alert(res.summary_text);
            
            // 或者填入报表单元格
            _g().getWidgetByName("cell_name").setValue(res.summary_text);
        }
    }
});
```

### 方式二：iframe嵌入播放器

```javascript
// 在报表中嵌入音频播放器
var audioHtml = '<audio controls src="http://your-server:5000/audio/report_夜班.mp3"></audio>';
// 通过HTML组件或公式显示
```

### 方式三：帆软后台配置回调

1. 进入帆软决策平台 → 服务器 → 数据连接
2. 配置HTTP数据集指向本服务的API地址
3. 在报表中通过数据集字段展示结果

## 🔧 配置详解

### LLM模型选择推荐

| 场景 | 推荐模型 | 说明 |
|------|----------|------|
| 性价比首选 | `gpt-4o-mini` | 成本低，效果不错 |
| 最佳效果 | `gpt-4o` | 质量最高 |
| 国内合规 | `qwen-plus` / `deepseek-chat` | 通过中转API使用 |
| 企业内部部署 | 自建模型 | 部署在本地服务器 |

### TTS语音引擎对比

| 引擎 | 成本 | 质量 | 说明 |
|------|------|------|------|
| Edge TTS | 免费 | ⭐⭐⭐⭐ | **推荐**，无需Key，微软神经网络声音 |
| Azure TTS | 付费 | ⭐⭐⭐⭐⭐ | 效果最佳，需订阅 |
| pyttsx3 | 免费 | ⭐⭐ | 离线备用，质量一般 |

## 📝 示例输出

> **夜班整体运行平稳。重点提示：纯化3号层析柱收率较前日下降 1.5%，为92.5%，但仍高于90%警戒线。柱效压力略有上升至1.8MPa，建议白班关注层析柱状态。其余批次检验项目均符合标准规定，无重大偏差。**

## ⚠️ 注意事项

1. **AI只做"翻译"和"朗读"，不做判断** —— 责任主体仍是人
2. **数据来源**：所有数据均来自帆软服务器，不支持模拟/演示数据
3. **降级机制**：帆软API不可用时自动降级到Playwright浏览器自动化，LLM不可用时使用规则模板
4. **安全建议**：生产环境请修改默认端口和账号密码

## 🐛 故障排查

| 问题 | 解决方案 |
|------|----------|
| LLM返回空内容 | 检查 API Key 是否有效，查看日志中的详细错误 |
| 语音无法播放 | 确认安装了 edge-tts (`pip install edge-tts`) |
| 无法连接帆软 | 检查 FINEREPORT_BASE_URL 和网络连通性 |
| 中文乱码 | 确保系统编码为 UTF-8 |

## 📄 License

MIT License - 内部项目使用
