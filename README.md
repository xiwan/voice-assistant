# voice-assistant

macOS / Windows / Linux 上的本地语音助手：说出唤醒词，语音指令自动转文字并交给
[kiro-cli](https://kiro.dev) 执行。纯 Rust 单二进制，推理全部本地运行，无云端依赖。

```
麦克风 ──► 唤醒词检测 ──► VAD 断句录音 ──► 语音转文字 ──► ACP agent ──► 流式 TTS
 cpal      openWakeWord      Silero VAD      whisper.cpp     常驻进程        macOS say
           (onnxruntime)    (onnxruntime)   (macOS Metal 加速)  (JSON-RPC/stdio)  (可换 Piper)
```

后端通过 **ACP（Agent Client Protocol）** 对接：agent 进程只启动一次并常驻，多轮
指令复用同一会话——既有上下文连续性，又省掉每轮冷启动。ACP 是通用协议，换成任何
会说 ACP 的 agent 只需改一行配置（`agent_cmd`），无需改代码。

agent 由一个独立的 **supervisor 线程**托管：主循环永不阻塞，任务执行时也一直在听，
可随时用"（唤醒词）停"打断（ACP `session/cancel`）；换个说法即自动打断当前、改做新的。
supervisor 保证任何时刻**有且只有一个** agent 存活——崩溃/卡死会自动重启（指数退避），
且绝不残留僵尸进程。

对话**不随进程走**：每轮说了什么都落盘（`~/.voice-assistant/session.json`），重连时先试
ACP `session/load` 把 agent 自己的会话接回来；接不回来（崩溃留下的会话锁、后端不支持、
或换了另一个 agent）就把对话摘要作为前缀挂在下一句指令上。所以关掉重开、agent 崩掉、
甚至中途换后端，都能接着聊；接回的方式（协议级 / 摘要）会明确告诉你，不含糊。
想从头开始就说"新会话"。

回答会**边生成边念**：文字一到句子结尾就送去合成，不等整段回答写完，所以说话节奏跟着
输出走。只念"它想说的话"——思考过程、工具调用、代码块、表格、URL 都不念。念的时候
麦克风**不静音**、始终在听，所以随时能喊"（唤醒词）停"掐断（外放且无 AEC 时可能听见
自己的声音，戴耳机最稳）。

## 特性

- 全本地推理：唤醒词 / VAD / ASR / TTS 都在本机跑，语音数据不出机器
- 单二进制部署：`cargo build` 出一个可执行文件，拷走即用
- 首次运行交互式设置：唤醒词、识别语言、模型精度、agent 权限、语音回复，配置持久化
- 模型按需自动下载（首次约 145MB–1.5GB，取决于所选 whisper 模型）
- 中文友好：简体输出引导、中文音色、hf-mirror 下载源
- 流式语音回复：按句边生成边念，不等整段答案；只念回答本身，思考/工具/代码块不念
- 会话连续性：ACP agent 常驻，多轮指令共享同一会话上下文，且无每轮冷启动
- 记忆不随进程走：对话落盘，重连优先 `session/load` 接回原会话，接不回则用摘要续接；换后端也能接着聊
- 主循环不阻塞：任务执行时也在听，随时可"（唤醒词）停"打断，或直接改口下达新指令
- 打断可续接：说"暂停/等等"先停下、去做别的，再说"继续"接着原任务干（同会话记忆）
- 监督型单例：supervisor 线程托管 agent，崩溃/卡死自动重启（指数退避），有且只有一个、无僵尸
- 执行过程可见：agent 思考、工具调用、授权结果都实时打印，长任务不再像卡死
- 追问免唤醒：回答后进入追问窗口，直接接着说即可多轮对话，静默超时自动播报下线
- 人格绑定唤醒词：连接的 agent 自动以唤醒词为名（如 "Jarvis"），自我介绍即是该名字
- kiro-cli 权限三档：只读 / 安全命令白名单 / 完全信任

## 依赖

- Rust 工具链（1.80+）
- cmake（编译 whisper.cpp 用）：`brew install cmake` / `apt install cmake` /
  Windows 用 [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/)（含 MSVC + cmake）
- Linux 额外需要 ALSA 头文件：`apt install libasound2-dev`
- [kiro-cli](https://kiro.dev) 已安装并登录
- macOS：首次运行需给终端授予麦克风权限（系统设置 → 隐私与安全性 → 麦克风）

三平台的构建与单元测试由 GitHub Actions 矩阵持续验证（`.github/workflows/ci.yml`）；
作者的日常开发机是 macOS，Windows / Linux 的**运行时**行为尚未实机验证。

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

### 桌面窗口

```bash
voice-assistant gui
```

跟终端模式跑的是**同一条流水线**——窗口只是订阅了同一份事件流的另一个前端，语音和界面
按钮走同一条派发路径，不会出现两套行为。窗口里能看到：

- 状态灯（待机 / 在听 / 思考中 / 重启中）和当前唤醒词
- 唤醒得分条 + 阈值刻度：每个音频窗口都画，阈值设得不对一眼就看出来
- 对话流：你说的话、流式生成的回答；思考过程和工具调用默认折叠，勾选后展开
- 暂停 / 继续 / 放弃三个按钮，和说"暂停/继续/算了"完全等价
- 输入框：不方便说话时直接打字
- ⚙ 设置面板：切 agent、装缺失的 CLI/适配器、拖参数、换监听方式、填模型 API key。
  每个 agent 前面有个手绘小标记（kiro 是幽灵、dsh 是鲸鱼、Claude 是放射线、Codex 是六边形）——
  不是厂商官方 logo：egui 自带字体没有这些 emoji 字形，而打包商标素材要引入图片解码器

用 egui + glow（OpenGL）而不是 Tauri：不引入 WebView，三平台一份代码。egui 自带字体
不含中文，程序会自动探测系统中文字体（macOS Hiragino/PingFang、Windows 微软雅黑、
Linux Noto Sans CJK / 文泉驿）；都找不到时会提示，界面中文会显示为方块。

### 怎么开始听：常听 or 按住说话

默认是**常听**：唤醒词一直在监听。设置面板里可切成**按住说话**（push-to-talk），
默认按 `Space`，可改键：

- 按住期间录音，**松开即结束**——不跑 VAD 断句，你的手指就是断句点，
  跑 VAD 会在你中途停顿时把话掐断。`max_utterance_ms` 仍然兜底防按键卡住，
  短于 250ms 的当误触丢弃
- 只在**窗口有焦点**时有效。全局热键（在别的 App 里也能按）需要新依赖和
  macOS 辅助功能授权，还没做
- 输入框获得键盘焦点时忽略该键，否则打字会一路发送出去

配置项是 `listen_mode`（wake / ptt）和 `ptt_key`。

### 模型 API key

kiro-cli / Claude Code / Codex 都用各自 CLI 的登录态，本程序不碰凭据。
DeepSeek Harness 需要 API key，在设置面板「模型凭据」里填。存放规则：

- 存在 **`~/.voice-assistant/secrets`**（权限 `0600`），**不在** `config` 里——
  这样你把配置贴进 issue 不会连带泄露；也不在仓库里，提交不进去
- 以**子进程环境变量**交给 agent，不进命令行参数，所以不会出现在日志、
  报错信息或 `ps` 输出里；只传当前 agent 声明的那一个变量
- 面板只显示掩码（`sk-…218（35 位）`），密码框不回填明文

### 打断与继续

任务执行中主循环一直在听，念唤醒词即可插话，按你说的话分流：

| 你说 | 触发词 | 行为 |
|------|--------|------|
| 暂停 | 暂停 / 等等 / 等一下 / 稍等 / 停一下 / 停 | 停下当前任务并**记住**它；随后可去做别的 |
| 继续 | 继续 / 接着 / 接下去 | 接着刚才被暂停的任务干（重述原指令，凭会话记忆续接）|
| 放弃 | 算了 / 取消 / 不用了 / 别做了 | 停下并**丢弃**记忆，不再续接 |
| 新会话 | 新会话 / 重新开始 / 清空上下文 / 忘掉刚才 | 忘掉整段对话，开一个干净的会话 |
| 其它 | —— | 当作新指令（正在忙则自动打断改做新的）|

"暂停"记住的任务是**粘性**的：中间穿插别的指令也不会丢，直到你"继续"或"放弃"。
底层用 ACP `session/cancel` 停当前轮，会话上下文保留，所以能续上。

### 记忆与会话恢复

每轮对话（你说的 + 助手答的）都写进 `~/.voice-assistant/session.json`（权限 `0600`，
只保留最近 40 轮、每轮上限 1200 字）。连上 agent 时按这个顺序续接：

1. **协议级接回** —— 同一个后端、且它声明支持 `loadSession` 时，发 ACP `session/load`
   把原会话接回来，agent 连自己当时的推理都还在。加载时 agent 会重放历史，这些重放
   **不打印也不朗读**（否则等于把整段对话重新念一遍）
2. **摘要续接** —— 接不回来时（崩溃留下的会话锁、后端不支持、或你换了别的 agent），
   开新会话并把对话摘要作为前缀挂在你下一句指令前面。不额外占一个回合，agent 也不会
   多说话；代价是它只知道对话线索，不知道自己当时怎么想的
3. **干净开始** —— 没有可续的东西，或你明确说了"新会话"

用哪种方式会直接告诉你（终端一行提示，窗口状态栏一个标记）。**这里有个坑值得知道**：
agent 进程被 SIGKILL 掉时，会话锁会留在原地且永不过期，那个会话就再也 load 不回来了——
所以替换 agent 时我们先关 stdin 让它自己退出，只有超时才杀。这也是为什么即便能 `session/load`
也照样留一份自己的摘要。

## 命令

| 命令 | 说明 |
|------|------|
| `voice-assistant` | 运行完整流水线（首次自动进入 setup） |
| `voice-assistant gui` | 桌面窗口（同一条流水线，多一个可见的界面） |
| `voice-assistant setup` | 重新设置：唤醒词 / 语言 / whisper 模型 / agent 后端 / 运行参数 |
| `voice-assistant devices` | 列出音频输入设备 |
| `voice-assistant test-wake` | 唤醒词调试：实时打印检测得分 |
| `voice-assistant test-vad` | VAD 调试：实时打印语音概率 |
| `voice-assistant test-asr` | 录一句话并打印转写结果 |
| `voice-assistant test-tts` | 语音回复调试：模拟流式回答并朗读（`--interrupt` 试打断） |
| `voice-assistant selftest` | 无麦克风自检（合成音频过一遍全部组件） |
| `voice-assistant session-test` | 会话恢复自检：断开重连后确认上下文还在（需先设 `VA_SESSION_FILE`，避免覆盖真实对话） |

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
| `tts` | `VA_TTS` | 按平台 | 语音回复引擎：say（macOS）/ sapi（Windows）/ espeak（Linux）/ cmd / off |
| `tts_voice` | `VA_TTS_VOICE` | 自动 | 音色，留空则按语言选（zh → Tingting / cmn） |
| `tts_rate` | `VA_TTS_RATE` | 0 | 语速（字/分钟），0 = 引擎默认 |
| `tts_cmd` | `VA_TTS_CMD` | — | tts=cmd 时的 sidecar 命令（从 stdin 读文本、自己合成并播放）|
| `silence_ms` | `VA_SILENCE_MS` | 1000 | 说完停顿多久算结束 |
| `no_speech_ms` | `VA_NO_SPEECH_MS` | 6000 | 唤醒后 / 追问窗口内不说话多久放弃（超时播报下线） |
| `max_utterance_ms` | `VA_MAX_UTTERANCE_MS` | 30000 | 单次录音上限 |
| — | `VA_MODELS_DIR` | ~/.voice-assistant/models | 模型目录 |
| — | `VA_HF_BASE` | https://hf-mirror.com | HuggingFace 下载源 |

\* `VA_WHISPER_MODEL` 直接指定 ggml 模型文件路径，跳过自动下载。

### 唤醒词

内置四个 openWakeWord 预训练唤醒词：Hey Jarvis / Alexa / Hey Mycroft / Hey Rhasspy。
也可以在 setup 中填入自己[训练的 openWakeWord 模型](https://github.com/dscripka/openWakeWord#training-new-models)（.onnx 路径）。

### 语音回复

setup 第 5 步开关，默认使用**系统自带**引擎，零额外安装：

| 平台 | 引擎 | 说明 |
|------|------|------|
| macOS | `say` | 系统内置，中文默认 Tingting |
| Windows | `sapi` | 经 PowerShell 调用 System.Speech，系统自带 |
| Linux | `espeak` | 仅在 PATH 上有 `espeak-ng` 时启用，否则默认关闭 |

Windows / Linux 的引擎命令行由代码构造（不经过配置文件的空格分词），要念的文本一律
走 stdin 而非命令行参数——所以回答里出现什么字符都不会变成 shell/PowerShell 代码。
这两个引擎音质一般且**尚未在实机验证**，追求自然度请用下面的 `tts=cmd` 接 Kokoro/Piper。

- **边生成边念**：回答文字一到句子结尾（。！？；或换行）就立刻送去合成；长句子超过
  48 字还没标点，会在逗号处先切一段出来念，所以开口时机跟着生成走，不用等整段写完。
- **只念该念的**：只有回答正文进 TTS。思考过程、工具调用、代码块、表格、分隔线一律
  跳过，行内的 markdown 标记会去掉，URL 念成"链接"。
- **朗读中可打断**：麦克风始终实时监听，念回答时喊"（唤醒词）停/暂停"即可掐断（`say`
  进程被 kill），说新指令也会立刻改做新的。戴耳机最稳；外放且无 AEC 时，助手的声音会
  混进麦克风，可能影响唤醒识别（下一步 Kokoro sidecar + 回声消除解决）。
- 回答刚生成完、还在念最后一句的那段"尾巴"里暂不响应唤醒打断，等这句念完即进入追问窗口。

音色用 `say -v '?'` 看全部；中文默认 Tingting。若嫌 `say` 中文音质一般，setup 里可先
换 macOS **增强中文音色**（系统设置 → 辅助功能 → 朗读内容 → 系统嗓音 → 管理嗓音）。

#### 换更好的引擎：`tts=cmd` sidecar

想要更自然的中文（如 **Kokoro** 神经 TTS）时，用 `cmd` 引擎：setup 第 5 步选"自定义
命令"，或设 `VA_TTS=cmd` + `VA_TTS_CMD="python3 /abs/path/scripts/kokoro_say.py"`。
约定很简单——sidecar **从 stdin 读一句文本、自己合成并播放、放完退出**（单进程，这样
"停"能把它 kill 掉打断）。因为是独立进程，Kokoro 自带的 onnxruntime 不会和本程序钉死的
`ort` 版本冲突。参考实现见 `scripts/kokoro_say.py`（需 `pip install kokoro-onnx
misaki[zh] sounddevice` 并下载中文模型）。同理可接 Piper 或任何别的 TTS。

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
- **听不到语音回复**：`voice-assistant test-tts` 直接试；`voice-assistant selftest` 会
  打印当前引擎和平台默认值（Linux 需先装 `espeak-ng`，或用 `tts=cmd` 接 piper/kokoro）
- **助手把自己的话当指令**：麦克风现在全程实时监听（为支持朗读中打断），只对唤醒词响应；
  外放且无 AEC 时若被自身声音误触，戴耳机可根治，或临时把 `tts` 调低音量/关掉
- **agent 起不来 / 后端报错看不到**：完整 stderr 在 `~/.voice-assistant/agent.log`，
  界面上只显示第一条有用的行。连续失败三次会自动改用别的可用 agent 并说明原因
- **agent 响应慢**：默认走 `kiro-cli acp --agent voice`（ACP 常驻进程，无 MCP 冷启动、多轮不重启）；`setup` 会自动配置
- **模型下载慢/失败**：`VA_HF_BASE` 换源，或手动下载放入模型目录

## Roadmap

- [x] 跨平台桌面窗口 — egui + glow，与终端共用同一份事件流（`voice-assistant gui`）
- [x] TTS 语音回复 — 流式按句朗读，macOS `say`
- [x] 朗读中语音打断 — 麦克风实时监听，唤醒词旁路（外放全双工需 AEC，见下）
- [x] 可换 TTS 引擎 — `tts=cmd` sidecar（接 Kokoro/Piper 等，独立进程绕开 ort 冲突）
- [ ] Kokoro 中文神经 TTS 参考 sidecar 打磨 + 外放回声消除（AEC，用自合成信号做参考）
- [x] 会话连续性（多轮对话保持上下文）— 常驻 ACP agent，且跨崩溃/重启/换后端也能续：
      `session/load` 接回，接不回用落盘摘要续接
- [ ] 更多 agent 后端（ACP 通用协议，改 `agent_cmd` 即可接入其它 ACP agent）
- [ ] 中文自定义唤醒词
- [ ] 危险操作语音确认

## License

MIT
