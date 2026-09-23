// SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Vaara governance for OpenCode. Installed by `vaara init`; removed by
// `vaara ungovern`. Every tool call, built-in or MCP, goes to
// `vaara hook pre-tool-use --client opencode` before it runs. Exit 0 lets
// it run. Any other result stops it: exit 2 is a Vaara verdict, and an
// engine that cannot run is not a gate that can pass anything, so that
// stops the call too unless "fail_open": true is set in
// ~/.vaara/claude-code/config.json.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const VAARA_BIN = process.env.VAARA_BIN || "__VAARA_BIN__";

function failOpen() {
  try {
    const cfg = JSON.parse(readFileSync(join(homedir(), ".vaara", "claude-code", "config.json"), "utf8"));
    return cfg && cfg.fail_open === true;
  } catch {
    return false;
  }
}

function runHook(subcommand, payload) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(VAARA_BIN, ["hook", subcommand, "--client", "opencode"], {
        env: { ...process.env, VAARA_PLUGIN_AGENT_ID: process.env.VAARA_PLUGIN_AGENT_ID || "opencode" },
        stdio: ["pipe", "ignore", "pipe"],
      });
    } catch (err) {
      resolve({ code: null, stderr: String(err) });
      return;
    }
    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (err) => resolve({ code: null, stderr: String(err) }));
    child.on("close", (code) => resolve({ code, stderr }));
    child.stdin.on("error", () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

export const VaaraGovernance = async () => ({
  "tool.execute.before": async (input, output) => {
    const { code, stderr } = await runHook("pre-tool-use", {
      tool: input.tool, sessionID: input.sessionID, callID: input.callID, args: output.args,
    });
    if (code === 0) return;
    const said = stderr.trim().split("\n").filter(Boolean).pop() || "";
    if (code === 2) throw new Error(said || `vaara-governance: BLOCKED ${input.tool}`);
    if (failOpen()) return;
    throw new Error(
      `vaara-governance: BLOCKED ${input.tool} (fail-closed): the Vaara engine at ${VAARA_BIN} ` +
      `could not decide (${said || `exit ${code}`}). Reinstall vaara and re-run \`vaara init\`, ` +
      `or set "fail_open": true in ~/.vaara/claude-code/config.json.`,
    );
  },
  "tool.execute.after": async (input, output) => {
    await runHook("post-tool-use", {
      tool: input.tool, sessionID: input.sessionID, callID: input.callID, args: input.args,
      output: { title: output && output.title, metadata: output && output.metadata },
    });
  },
});
