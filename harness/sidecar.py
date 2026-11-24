"""Client of the Node sidecar (bridge/sidecar.mjs): legacy layout / equipment, legacy's own prep-start focus, the bounty
kill chance. One persistent `node` process, one JSON line per request; `request` raises SidecarError on {"ok": false}.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = os.environ.get("ARK_ENGINE", "E:/ArkHarness-engine/sp-label")


class SidecarError(RuntimeError):
    pass


class Sidecar:
    def __init__(self, engine: str = ENGINE):
        self.p = subprocess.Popen(["node", str(ROOT / "bridge/sidecar.mjs"), engine], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, cwd=str(ROOT))
        self.lock = threading.Lock()
        self.seq = 0
        self.ms: dict[str, list] = {}

    def request(self, op: str, state: dict, **kw) -> dict:
        with self.lock:
            self.seq += 1
            rid = self.seq
            t0 = time.perf_counter()
            self.p.stdin.write(json.dumps({"id": rid, "op": op, "state": state, **kw}, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            if not line:
                raise SidecarError(f"sidecar exited (code {self.p.poll()})")
            r = json.loads(line)
            self.ms.setdefault(op, []).append((time.perf_counter() - t0) * 1000)
        if r.get("id") != rid:
            raise SidecarError(f"response id {r.get('id')} != {rid}")
        if not r.get("ok"):
            raise SidecarError(f"{op}: {r.get('error')}")
        return r

    def close(self):
        try:
            self.p.kill()
        except OSError:
            pass
