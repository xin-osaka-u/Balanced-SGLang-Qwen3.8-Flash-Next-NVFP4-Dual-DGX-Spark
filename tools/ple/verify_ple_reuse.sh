#!/usr/bin/env bash
# Check the on-disk PLE tables after a reuse boot, on every node.
#   - table size / mtime (mtime must NOT advance while serving: read-only mapping)
#   - sidecar fingerprint present and complete
#   - optional --hash: sha256 of each 25.6 GB table (~1-2 min each, run when idle);
#     compare hashes before and after serving requests.
# For a byte-exact check against the checkpoint use full_compare.py on each node.
#
# usage: HOSTS="spark1 spark2" PLE_DIR=~/.cache/sglang/ple/<model-key> verify_ple_reuse.sh [--hash]
set -u
HASH=0; [ "${1:-}" = "--hash" ] && HASH=1
: "${PLE_DIR:?set PLE_DIR to the directory holding ple_table_*.bin}"
for h in ${HOSTS:-localhost}; do
  echo "===== $h"
  ssh -o BatchMode=yes "$h" "cd '$PLE_DIR' 2>/dev/null || { echo 'no PLE dir'; exit 0; }
    for f in ple_table_*.bin; do
      echo \"table: \$f size=\$(stat -c %s \$f) mtime=\$(stat -c %y \$f | cut -c1-19)\"
      if [ -f \$f.meta.json ]; then
        python3 -c \"import json;m=json.load(open('\$f.meta.json'));fp=m['fingerprint'];print('sidecar: complete=%s rank=%s tp=%s sglang=%s'%(m.get('complete'),fp.get('tp_rank'),fp.get('tp_size'),fp.get('sglang_version')))\"
      else echo 'sidecar: MISSING (next boot rebuilds)'; fi
      if [ $HASH = 1 ]; then echo -n 'sha256: '; sha256sum \$f | cut -c1-16; fi
    done
    grep -l . /proc/\$(pgrep -f sglang.launch_server | head -1)/maps >/dev/null 2>&1 && \
      grep ple_table /proc/\$(pgrep -f sglang.launch_server | head -1)/maps | awk '{print \"mapping perms:\", \$2}' | sort -u"
done
