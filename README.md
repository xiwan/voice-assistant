# voice-assistant

macOS / Linux 上的本地语音助手：说出唤醒词，语音指令自动转文字并交给
[kiro-cli](https://kiro.dev) 执行。纯 Rust 单二进制，推理全部本地运行，无云端依赖。

```
麦克风 ──► 唤醒词检测 ──► VAD 断句录音 ──► 语音转文字 ──► ACP agent
 cpal      openWakeWord      Silero VAD      whisper.cpp     常驻进程 (JSON-RPC/stdio)
           (onnxruntime)    (onnxruntime)   (macOS Metal 加速)  默认 kiro-cli acp
```

后端通过 **ACP（Agent Client Protocol）** 对接：agent 进程只启动一次并常驻，多轮
指令复用同一会话——既有上下文连续性，又省掉每轮冷启动。ACP 是通用协议，换成任何
会说 ACP 的 agent 只需改一行配置（`agent_cmd`），无需改代码。

## 特性

- 全本地推理：唤醒词 / VAD / ASR 都在本机跑，语音数据不出机器
- 单二进制部署：`cargo build` 出一个可执行文件，拷走即用
- 首次运行交互式设置：唤醒词、识别语言、模型精度、agent 权限，配置持久化
- 模型按需自动下载（首次约 145MB–1.5GB，取决于所选 whisper 模型）
- 中文友好：简体输出引导、hf-mirror 下载源
- 会话连续性：ACP agent 常驻，多轮指令共享同一会话上下文，且无每轮冷启动
- 追问免唤醒：回答后进入追问窗口，直接接着说即可多轮对话，静默超时自动播报下线
- 人格绑定唤醒词：连接的 agent 自动以唤醒词为名（如 "Jarvis"），自我介绍即是该名字
- kiro-cli 权限三档：只读 / 安全命令白名单 / 完全信任

## 依赖

- Rust 工具链（1.80+）
- cmake（编译 whisper.cpp 用）：`brew install cmake` / `apt install cmake`
- [kiro-cli](https://kiro.dev) 已安装并登录
- macOS：首次运行需给终端授予麦克风权限（系统设置 → 隐私与安全性 → 麦克风）

## 快速开始

```bash
git clone https://github.com/xiwan/voice-assistant.git
cd voice-assistant
cargo build --release

# 首次运行进入交互式设置，之后自动下载所需模型
./target/release/voice-assistant
```

说出唤醒词（默认 "Hey Jarvis"），听到提示后说出指令，说完停顿 1 秒自动结束，
识别文本会交给 agent，回答流式打印到终端。回答后进入**追问窗口**：直接接着说即可
继续对话（无需再念唤醒词，共享同一会话上下文）；`no_speech_ms` 内没开口，助手会
播报一句下线提示并回到等唤醒词状态。

## 命令

| 命令 | 说明 |
|------|------|
| `voice-assistant` | 运行完整流水线（首次自动进入 setup） |
| `voice-assistant setup` | 重新设置：唤醒词 / 语言 / whisper 模型 / agent 后端 / 运行参数 |
| `voice-assistant devices` | 列出音频输入设备 |
| `voice-assistant test-wake` | 唤醒词调试：实时打印检测得分 |
| `voice-assistant test-vad` | VAD 调试：实时打印语音概率 |
| `voice-assistant test-asr` | 录一句话并打印转写结果 |
| `voice-assistant selftest` | 无麦克风自检（合成音频过一遍全部组件） |

## 配置

配置文件：`~/.voice-assistant/config`（key=value），环境变量优先级更高。

| 配置项 | 环境变量 | 默认 | 说明 |
|--------|---------|------|------|
| `wake_word` | — | hey_jarvis | 唤醒词 id 或自定义 .onnx 路径 |
| `lang` | `VA_LANG` | auto | 识别语言（zh 时自动引导简体输出） |
| `whisper` | `VA_WHISPER_MODEL`* | base | whisper 模型：base / small / medium |
| `threshold` | `VA_WAKE_THRESHOLD` | 0.5 | 唤醒词检测阈值 |
| `agent_mode` | — | readonly | kiro 权限：readonly / safe / full |
| `agent_cmd` | `VA_AGENT_CMD` | kiro-cli acp --agent voice | ACP agent 启动命令（换后端只改这里） |
| `silence_ms` | `VA_SILENCE_MS` | 1000 | 说完停顿多久算结束 |
| `no_speech_ms` | `VA_NO_SPEECH_MS` | 6000 | 唤醒后 / 追问窗口内不说话多久放弃（超时播报下线） |
| `max_utterance_ms` | `VA_MAX_UTTERANCE_MS` | 30000 | 单次录音上限 |
| — | `VA_MODELS_DIR` | ~/.voice-assistant/models | 模型目录 |
| — | `VA_HF_BASE` | https://hf-mirror.com | HuggingFace 下载源 |

\* `VA_WHISPER_MODEL` 直接指定 ggml 模型文件路径，跳过自动下载。

### 唤醒词

内置四个 openWakeWord 预训练唤醒词：Hey Jarvis / Alexa / Hey Mycroft / Hey Rhasspy。
也可以在 setup 中填入自己[训练的 openWakeWord 模型](https://github.com/dscripka/openWakeWord#training-new-models)（.onnx 路径）。

### Agent 后端

setup 第 4 步选择后端：默认 **kiro-cli**（自动生成并管理权限文件），也可选
**自定义 ACP 命令**，填入本机任意会说 ACP 的 agent 启动命令（例如 `my-agent acp`）。
ACP 是通用协议，切换后端不需要改代码。自定义后端会询问是否自动批准工具授权（语音
场景无法交互确认），也可随时直接编辑 config 里的 `agent_cmd`。

对 kiro-cli 后端，助手会以唤醒词为 agent 的名字（如 "Hey Jarvis" → agent 自称
"Jarvis"）：启动时按当前唤醒词自动重写 `~/.kiro/agents/voice.json` 的身份 prompt，
所以连接的 kiro 就等同于你唤醒的那个人格。自定义后端的人格由其自身管理。

### kiro-cli 权限模式

setup 会按所选模式生成 `~/.kiro/agents/voice.json`（该文件由 voice-assistant 托管：
setup 重设或每次启动都会按当前唤醒词/权限重写，请勿手改）：

- **readonly** — 只能读文件，最安全
- **safe** — 额外放行只读命令白名单（pwd / ls / cat / git status 等），
  白名单外的命令在非交互模式下一律拒绝
- **full** — 任意命令 + 写文件。语音识别可能听错，慎用

## 模型来源

首次运行自动下载到 `~/.voice-assistant/models/`：

| 模型 | 来源 | 大小 |
|------|------|------|
| melspectrogram / embedding / 唤醒词 | [openWakeWord releases](https://github.com/dscripka/openWakeWord/releases) | ~3.5MB |
| Silero VAD v5 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | 2.2MB |
| whisper ggml | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)（经 hf-mirror） | 141MB–1.5GB |

## 故障排查

- **听不到唤醒**：`voice-assistant test-wake` 看得分，长期低于 0.5 可降低 `threshold`
- **没有输入设备**：确认终端有麦克风权限；`voice-assistant devices` 列出设备
- **中文识别差**：`setup` 换 small/medium 模型
- **agent 响应慢**：默认走 `kiro-cli acp --agent voice`（ACP 常驻进程，无 MCP 冷启动、多轮不重启）；`setup` 会自动配置
- **模型下载慢/失败**：`VA_HF_BASE` 换源，或手动下载放入模型目录

## Roadmap

- [ ] TTS 语音回复
- [x] 会话连续性（多轮对话保持上下文）— 通过常驻 ACP agent 实现
- [ ] 更多 agent 后端（ACP 通用协议，改 `agent_cmd` 即可接入其它 ACP agent）
- [ ] 中文自定义唤醒词
- [ ] 危险操作语音确认

## License

MIT
