#!/usr/bin/env bash
#
# restart.sh 的执行体。不要直接调用，由 restart.sh detach 后台启动。
#
# 用法: _restart_helper.sh <mode: terminal|detach> <grace_seconds> <pid> [pid...]
#
# 它会活到父进程链全部死掉之后（restart.sh 用 nohup 放飞，SIGHUP 被挡，
# 孤儿进程由 launchd 收养），所以能在杀掉主进程之后再把它拉起来。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# VA_RESTART_BIN 仅用于演练：可指向无害命令，验证 kill/清理/拉起三步而不真动助手。
BIN="${VA_RESTART_BIN:-$REPO/target/release/voice-assistant}"
LOG="$HOME/.voice-assistant/run.log"
TERM_TIMEOUT=5

mode="$1"; grace="$2"; shift 2
pids=("$@")

log() { echo "[helper $(date +%T)] $*" >>"$LOG"; }

mkdir -p "$(dirname "$LOG")"
log "启动，模式=$mode 宽限=${grace}s 目标=${pids[*]}"

# 1. 留时间给 agent 把最后一句话说完
sleep "$grace"

# 2. 先温和后强硬
for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 "$TERM_TIMEOUT"); do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  [ "$alive" = 0 ] && break
  sleep 1
done
for pid in "${pids[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done
log "主进程已终止"

# 3. 主进程被信号杀死时 Rust 的 Drop 不会执行，可能残留 agent 子进程。
#    它们的 stdin 会因父进程退出而 EOF、通常自己会走，这里兜一手。
#    只清理已经变成孤儿（ppid=1）的，避免误杀仍挂在别的活主进程下的 agent。
sleep 1
leftover=$(ps -Ao pid=,ppid=,command= | awk '$2==1 && /acp --agent voice/ && !/awk/ {print $1}')
if [ -n "$leftover" ]; then
  log "清理残留孤儿 agent: $(echo "$leftover" | tr '\n' ' ')"
  echo "$leftover" | xargs kill -TERM 2>/dev/null || true
fi
sleep 1

# 4. 拉起新的主进程
if [ "$mode" = terminal ]; then
  # 新开一个 Terminal.app 窗口：有真正的 TTY，麦克风权限仍挂在 Terminal 上，
  # 输出用户能直接看见。
  if osascript -e "tell application \"Terminal\" to do script \"cd $REPO && exec $BIN\"" >>"$LOG" 2>&1; then
    osascript -e 'tell application "Terminal" to activate' >>"$LOG" 2>&1 || true
    log "已在新 Terminal 窗口重启"
    exit 0
  fi
  # osascript 首次可能弹自动化授权框，没人点就会失败。绝不能停在
  # 「杀完了没人拉起来」，所以退回后台启动。
  log "osascript 失败（自动化权限？），退回后台启动"
fi

cd "$REPO" || exit 1
nohup "$BIN" >>"$LOG" 2>&1 &
log "已后台重启 (pid $!)，输出在 $LOG"
