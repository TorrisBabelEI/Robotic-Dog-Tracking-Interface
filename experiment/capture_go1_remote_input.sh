#!/usr/bin/env bash
# Operator-run Ubuntu helper: passive factory capture; no controller deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'from experiment.decode_native_go1_pcap import decode'
printf '%s\n' 'Factory-control diagnostic: robot already prone, belly and all four feet supported.' \
  'Keep the factory remote ON and connected. Leave sticks centered.' \
  'This records traffic and prompts one L2+B press; it starts no custom controller.'
read -r -p 'Confirm this posture and that no custom controller is running (type READY): ' answer
[[ "$answer" == READY ]] || { echo 'Cancelled before Pi access.'; exit 1; }
mkdir -p logs/remote-input-diagnostic
archive=$(mktemp -d "$PWD/logs/remote-input-diagnostic/review-XXXXXXXX")
printf 'archive=%s\n' "$archive"
remote=$(ssh -o ConnectTimeout=10 pi@192.168.12.1 \
  'mkdir -p /home/pi/go1-prone-engagement/logs && mktemp -d /home/pi/go1-prone-engagement/logs/remote-input-XXXXXXXX')
[[ "$remote" =~ ^/home/pi/go1-prone-engagement/logs/remote-input-[A-Za-z0-9]+$ ]] || { echo 'Unexpected remote path; stopping'; exit 1; }
printf '%s\n' "$remote" > "$archive/pi-directory.txt"
script=$(cat <<'PI'
set -euo pipefail
cd "$1"
sudo -v
ps -eo pid,comm,args > processes.txt
sudo ss -Huanp > sockets.txt
capture_pid=''
trap 'if [[ -n "$capture_pid" ]]; then wait "$capture_pid" || true; fi' EXIT
sudo timeout -s INT 20 tcpdump -i eth0 -nn -s 0 -U -w remote_input.pcap \
  'ip and udp and (port 8007 or port 8008)' >capture.txt 2>&1 &
capture_pid=$!
sleep 5
# sudo can make this PID root-owned: kill -0 then returns EPERM even
# while capture is running. Inspect process existence without signaling it.
if ! ps -p "$capture_pid" -o pid= >/dev/null; then
  set +e
  wait "$capture_pid"
  status=$?
  set -e
  capture_pid=''
  cat capture.txt
  printf 'STOP: capture exited before button prompts (status=%s).\n' "$status" >&2
  exit 1
fi
printf '\nNOW: hold factory L2+B together for two seconds. Keep sticks centered.\n'
date -u '+press_prompt=%Y-%m-%dT%H:%M:%SZ' > events.txt
sleep 2
printf '\nRELEASE both buttons now; leave the robot undisturbed.\n'
date -u '+release_prompt=%Y-%m-%dT%H:%M:%SZ' >> events.txt
set +e
wait "$capture_pid"
status=$?
set -e
capture_pid=''
cat capture.txt
[[ "$status" == 0 || "$status" == 124 ]] || exit "$status"
# Retain Pi originals; let pi read its new diagnostic files for download.
sudo chmod a+r remote_input.pcap
sha256sum remote_input.pcap > capture.sha256
printf '\nCapture complete.\n'
PI
)
# Quote the complete remote program as one Bash argument; no file deployment.
printf -v command 'bash -c %q -- %q' "$script" "$remote"
ssh -t -o ConnectTimeout=10 pi@192.168.12.1 "$command"
files=()
for name in remote_input.pcap capture.txt capture.sha256 events.txt processes.txt sockets.txt; do
  files+=("pi@192.168.12.1:$remote/$name")
done
scp "${files[@]}" "$archive/"
(cd "$archive" && sha256sum -c capture.sha256)
python3 -B experiment/decode_native_go1_pcap.py "$archive/remote_input.pcap" --out "$archive/decoded" \
  | tee "$archive/analysis.txt"
printf '\narchive=%s\n' "$archive"
printf '%s\n' 'Keep this archive. Do not start prone-engagement yet; review remote transitions and feedback first.'
