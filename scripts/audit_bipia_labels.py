"""Audit the BIPIA auto-labeller: how many `reflects_injection=True` rows
actually have signal carried by parameters, vs how many are example.com
placeholders the labeller false-positived?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _v039_common import load_bipia  # noqa: E402

TRACE_DIR = REPO / "tests/adversarial/traces"


def main() -> int:
    trials = {t.trial_id: t for t in load_bipia()}
    print(f"Loaded {len(trials)} BIPIA trials with injection_text.")

    url_in_inj = host_in_inj = example_com_only = 0
    shell_signal = file_signal = 0
    other = 0
    samples_example_com: list[dict] = []
    samples_other: list[dict] = []
    total_follows = 0
    total_follows_with_body = 0

    for f in sorted(TRACE_DIR.glob("bipia-s43-*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            trial = trials.get(d["trial_id"])
            if not trial:
                continue
            inj = trial.injection_text.lower()
            for tc in d.get("tool_calls", []):
                if not tc.get("reflects_injection"):
                    continue
                total_follows += 1
                tn = tc.get("tool_name", "")
                p = tc.get("parameters", {})
                if tn == "http_request":
                    url = str(p.get("url", "")).lower()
                    body = str(p.get("body", "")).strip()
                    if body:
                        total_follows_with_body += 1
                    host = ""
                    if "://" in url:
                        host = url.split("://", 1)[1].split("/", 1)[0]
                    matched = False
                    if url and url in inj:
                        url_in_inj += 1
                        matched = True
                    elif host and host != "example.com" and host in inj:
                        host_in_inj += 1
                        matched = True
                    if not matched and (host == "example.com" or host.endswith(".example.com")):
                        example_com_only += 1
                        matched = True
                        if len(samples_example_com) < 5:
                            samples_example_com.append({
                                "model": d["model"], "attack_class": d["attack_class"],
                                "url": url, "body": body[:80]})
                    if not matched:
                        other += 1
                        if len(samples_other) < 5:
                            samples_other.append({
                                "model": d["model"], "attack_class": d["attack_class"],
                                "url": url, "body": body[:80]})
                elif tn == "shell_exec":
                    cmd = str(p.get("command", "")).strip()
                    if cmd and cmd[:64].lower() in inj:
                        shell_signal += 1
                    else:
                        other += 1
                elif tn == "file_write":
                    content = str(p.get("content", "")).strip()
                    if content and content[:64].lower() in inj:
                        file_signal += 1
                    else:
                        other += 1

    print(f"\nTotal `reflects_injection=True` rows: {total_follows}")
    print(f"  url-substring-in-injection signal:  {url_in_inj}")
    print(f"  host-substring-in-injection signal: {host_in_inj}")
    print(f"  shell-command-prefix signal:        {shell_signal}")
    print(f"  file-content-prefix signal:         {file_signal}")
    print(f"  example.com placeholder ONLY (no other signal): {example_com_only}  "
          f"<-- LIKELY FALSE POSITIVES")
    print(f"  no detectable signal:               {other}")
    print(f"\nhttp_request follows with non-empty body: {total_follows_with_body}")
    print("\nSample example.com-only `follows`:")
    for s in samples_example_com:
        print(f"  {s['model']:24s} | {s['attack_class'][:30]:30s} | url={s['url']}")
    if samples_other:
        print("\nSample 'no signal' rows:")
        for s in samples_other:
            print(f"  {s['model']:24s} | {s['attack_class'][:30]:30s} | url={s['url']} body={s['body']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
