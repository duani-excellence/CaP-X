"""Stop only the service processes recorded during this setup, guarding PID reuse."""
import json
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parents[2]
records = json.loads((root / "runtime/started-services.json").read_text())
for record in records:
    pid = record["pid"]
    proc = Path(f"/proc/{pid}")
    try:
        started = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        same = started == record["start_ticks"] and (proc / "cwd").resolve() == root
    except (FileNotFoundError, ProcessLookupError):
        same = False
    if same:
        os.kill(pid, signal.SIGINT if record["name"] == "web" else signal.SIGTERM)
        print(f"Stopping {record['name']} (PID {pid})")
        for _ in range(50):
            if not proc.exists():
                break
            time.sleep(0.1)
    else:
        print(f"Skipping {record['name']}: original process is no longer running")
