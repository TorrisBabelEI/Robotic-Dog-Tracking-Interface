#!/usr/bin/env bash
# Run on Ubuntu. Restore only the existing vendor Programming Module on Pi.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/programming-restoration
archive=$(mktemp -d "$PWD/logs/programming-restoration/review-XXXXXXXX")
printf 'restoration_archive=%s\n' "$archive"
verify=$(cat <<'VERIFY'
pattern='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
mapfile -t pids < <(pgrep -f "$pattern")
[[ ${#pids[@]} -eq 1 ]] || { echo 'STOP: expected exactly one Programming Module'; exit 1; }
if [[ -n "${1:-}" && "${pids[0]}" != "$1" ]]; then
  echo 'STOP: module PID changed after SSH logout; inspect before proceeding'; exit 1
fi
pgrep -af "$pattern"
sockets=$(ss -Huanp)
owned=$(awk -v module_pid="${pids[0]}" '$4 ~ /:8090$/ && $5 == "192.168.123.161:8082" && index($0, "pid=" module_pid ",") { print }' <<< "$sockets")
[[ -n "$owned" ]] || { echo 'STOP: module does not own expected 8090 -> 8082 socket'; exit 1; }
printf '%s\n' "$owned"
printf 'verified_module_pid=%s\n' "${pids[0]}"
VERIFY
)
script=$(cat <<'PI'
set -euo pipefail
[[ "$(id -un)" == pi && -d /home/pi/Unitree/autostart/programming ]] || exit 1
if pgrep -af '^([^ ]*/)?go1_lowlevel_experiment( |$)'; then
  echo 'STOP: experiment still running; no factory module started'; exit 1
fi
pattern='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
mapfile -t pids < <(pgrep -f "$pattern")
case ${#pids[@]} in
  0)
    mkdir -p /home/pi/go1-prone-engagement/logs
    startup_log=$(mktemp /home/pi/go1-prone-engagement/logs/programming-restore-XXXXXXXX.log)
    # The vendor wrapper backgrounds Python. Ignore SIGHUP and detach all
    # terminal descriptors for the wrapper AND the Python child it starts.
    (cd /home/pi/Unitree/autostart/programming &&
      nohup bash ./programming.sh </dev/null >"$startup_log" 2>&1)
    printf 'Pi startup log: %s\n' "$startup_log"
    sleep 2
    cat "$startup_log"
    ;;
  1) echo 'Programming Module already present; not starting a duplicate' ;;
  *) echo 'STOP: multiple Programming Modules; inspect before proceeding'; exit 1 ;;
esac
PI
)
script+=$'\n'"$verify"
printf -v command 'bash -c %q' "$script"
ssh -t -o ConnectTimeout=10 pi@192.168.12.1 "$command" | tee "$archive/before-logout.txt"
pid=$(sed -n 's/^verified_module_pid=\([0-9][0-9]*\)\r*$/\1/p' "$archive/before-logout.txt")
[[ "$pid" =~ ^[0-9]+$ ]] || { echo 'STOP: no unique verified module PID'; exit 1; }
printf '%s\n' 'First SSH session closed; checking the same PID and socket in a fresh session.'
script=$'set -euo pipefail\nsleep 2\n'"$verify"
printf -v command 'bash -c %q -- %q' "$script" "$pid"
# Read-only follow-up: do not silently restart a process lost after logout.
ssh -o ConnectTimeout=10 pi@192.168.12.1 "$command" | tee "$archive/after-logout.txt"
printf 'programming_restoration=PASS; same PID and socket verified after SSH logout\n' | tee "$archive/result.txt"
