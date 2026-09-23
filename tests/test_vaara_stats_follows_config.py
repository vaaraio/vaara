"""/vaara-stats reads the trail the hooks write.

`vaara init` points the hooks at the unified trail through the `audit_db` key
in the plugin config. The stats script resolved only the environment override
and the legacy default, so after `vaara init` it read the old file and
reported it missing. It now uses the hooks' own resolution.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/claude-code-vaara-governance/scripts/vaara_stats.py"


def _trail(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("create table audit_records (record_id text, action_id text, "
                "event_type text, timestamp real, agent_id text, tool_name text, data text)")
    con.execute("insert into audit_records values ('r1','a1','decision_made',1.0,'x','Bash','{}')")
    con.commit()
    con.close()


def _run(home: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", **(env_extra or {})}
    return subprocess.run([sys.executable, str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=60)


def test_stats_follow_the_config_audit_db(tmp_path):
    unified = tmp_path / ".vaara" / "trail" / "audit.db"
    _trail(unified)
    cfg = tmp_path / ".vaara" / "claude-code" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"audit_db": str(unified)}))
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert f"audit_db: {unified}" in proc.stdout
    assert "records:  1" in proc.stdout


def test_the_environment_override_still_wins(tmp_path):
    other = tmp_path / "elsewhere.db"
    _trail(other)
    proc = _run(tmp_path, {"VAARA_PLUGIN_AUDIT_DB": str(other)})
    assert f"audit_db: {other}" in proc.stdout
