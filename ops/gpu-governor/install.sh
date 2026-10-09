#!/usr/bin/env bash
# Install or update the governor on THIS node. Needs sudo. Does not start it unless --start is given.
#   sudo ./install.sh master|worker [--start]
# Safe to re-run; keeps an existing /etc/gpu-governor.env unless --force-env.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NODE=${1:?usage: install.sh master|worker [--start] [--force-env]}; shift
START=0; FORCE=0
for a in "$@"; do case $a in --start) START=1;; --force-env) FORCE=1;; esac; done
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
install -d -m 755 /opt/gpu-governor /var/log/gpu-governor
install -m 755 "$HERE/gpu_governor.py" /opt/gpu-governor/gpu_governor.py
install -m 644 "$HERE/gpu-governor.service" /etc/systemd/system/gpu-governor.service
if [ ! -f /etc/gpu-governor.env ] || [ $FORCE = 1 ]; then
  install -m 600 "$HERE/gpu-governor.env.$NODE" /etc/gpu-governor.env
  echo "wrote /etc/gpu-governor.env from gpu-governor.env.$NODE (edit GOV_TOKEN, GOV_DRY_RUN)"
else
  echo "kept existing /etc/gpu-governor.env"
fi
systemctl daemon-reload
echo "installed. dry run: $(grep ^GOV_DRY_RUN /etc/gpu-governor.env)"
if [ $START = 1 ]; then systemctl enable --now gpu-governor; sleep 3; systemctl --no-pager status gpu-governor | head -12; fi
echo "status: curl -s http://$(grep ^GOV_LISTEN /etc/gpu-governor.env | cut -d= -f2)/status | python3 -m json.tool"
