#!/usr/bin/env bash
#
# 重启 voice-assistant 主进程，并保证对话记忆不丢。
#
# 为什么需要这个脚本：agent（kiro-cli）是 voice-assistant 的子进程，
# 自己杀自己没用（supervisor 会原地拉起一个新的，但配置是主进程启动时
# 读一次的）。要让配置生效必须重启主进程，而主进程一死 agent 也跟着死，
# 没人负责拉起来。所以由 detach 出去的 _restart_helper.sh 完成 kill + relaunch。
#
# 记忆怎么延续：agent 在调用本脚本之前，必须把对话摘要写进
#   ~/.kiro/steering/voice-memory.md
# kiro 每个新会话都会自动加载 ~/.kiro/steering/ 下的文件，所以重启后的
# agent 开口就带着上一轮的上下文。本脚本强制检查该文件存在且足够新，
# 否则拒绝重启 —— 宁可不重启，也不能把记忆丢了。
#
# 用法:
#   scripts/restart.sh --dry-run   只检查并打印将要做什么，不动任何进程
#   scripts/restart.sh             新开一个 Terminal 窗口重启（推荐）
#   scripts/restart.sh --detach    后台重启，输出写进 ~/.voice-assistant/run.log
#   scripts/restart.sh --force     跳过记忆文件检查（不推荐）
#
# 环境变量:
#   GRACE_BEFORE_KILL=15   kill 之前等几秒，留给 agent 把话说完（默认 3）

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/target/release/voice-assistant"
HELPER="$REPO/scripts/_restart_helper.sh"
MEMORY_FILE="$HOME/.kiro/steering/voice-memory.md"
LOG="$HOME/.voice-assistant/run.log"

# 记忆文件多久算「过期」（秒）。防止拿上一轮的旧摘要就重启。
MEMORY_MAX_AGE=600
GRACE_BEFORE_KILL="${GRACE_BEFORE_KILL:-3}"

mode=terminal
dry_run=0
force=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --detach)  mode=detach ;;
    --force)   force=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $arg（试试 --help）" >&2; exit 2 ;;
  esac
done

die() { echo "[restart] 中止: $*" >&2; exit 1; }

# ---- 1. 前置检查 ----
[ -x "$BIN" ] || die "找不到可执行文件 $BIN（先 cargo build --release）"
[ -x "$HELPER" ] || die "找不到 helper $HELPER（或缺少执行权限）"

# ---- 2. 记忆守卫：没有新鲜摘要就不许重启 ----
if [ "$force" = 1 ]; then
  echo "[restart] --force：跳过记忆文件检查"
else
  [ -s "$MEMORY_FILE" ] || die "记忆文件不存在或为空: $MEMORY_FILE
        重启会丢掉整段对话。请先把对话摘要写进该文件。"
  age=$(( $(date +%s) - $(stat -f %m "$MEMORY_FILE") ))
  [ "$age" -le "$MEMORY_MAX_AGE" ] || die "记忆文件已经 ${age}s 没更新（上限 ${MEMORY_MAX_AGE}s），
        可能是上一轮的旧摘要。请先刷新它，或明确加 --force。"
  echo "[restart] 记忆文件 OK（${age}s 前更新，$(wc -c <"$MEMORY_FILE" | tr -d ' ') 字节）"
fi

# ---- 3. 找到正在跑的主进程 ----
# 注意：这台机器上 pgrep 匹配不到它（实测返回空），所以用 ps + awk。
# 只认 argv[0] 的 basename 等于 voice-assistant，避免误杀本脚本或编辑器。
pids=$(ps -Ao pid=,command= | awk '{ c=$2; sub(/.*\//,"",c); if (c=="voice-assistant") print $1 }')
[ -n "$pids" ] || die "没找到正在运行的 voice-assistant 进程"
echo "[restart] 目标进程: $(echo "$pids" | tr '\n' ' ')"

# ---- 4. 交给 helper ----
if [ "$dry_run" = 1 ]; then
  echo "[restart] --dry-run，将要执行的是:"
  echo "    nohup $HELPER $mode $GRACE_BEFORE_KILL $(echo "$pids" | tr '\n' ' ') &"
  echo "[restart] （helper 会等 ${GRACE_BEFORE_KILL}s，SIGTERM→SIGKILL，清残留，再拉起新进程）"
  exit 0
fi

echo "[restart] ${GRACE_BEFORE_KILL}s 后杀掉主进程并重启（模式: $mode，日志: $LOG）"
# nohup 挡 SIGHUP；父进程链全死后 helper 被 launchd 收养，足以活到拉起新进程。
nohup "$HELPER" "$mode" "$GRACE_BEFORE_KILL" $pids >/dev/null 2>&1 &
echo "[restart] helper 已放飞 (pid $!)。再见，${GRACE_BEFORE_KILL}s 后见。"
