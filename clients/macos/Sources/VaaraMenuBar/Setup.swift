// Setup: the app wires the machine's AIs into Vaara itself.
//
// Scans known MCP client configs (Claude Desktop, Claude Code, Cursor,
// Windsurf), shows which servers are governed (routed through
// vaara-mcp-proxy) and which are naked, and rewrites the config on one
// click: each ungoverned entry becomes
//     vaara-mcp-proxy --upstream CMD --upstream-arg ARG ... --db TRAIL
// with a timestamped backup written next to the original first.
// Absolute proxy path is resolved at rewrite time because GUI apps
// launch with a minimal PATH.

import Foundation

struct EngineStatus {
    let vaaraPath: String?
    let proxyPath: String?
    var ok: Bool { vaaraPath != nil && proxyPath != nil }
}

struct MCPClient: Identifiable {
    let id: String          // display name
    let configPath: URL
    var governed: Int
    var ungoverned: Int
    var exists: Bool
    var hasBackup: Bool
}

enum SetupScanner {

    static let knownClients: [(name: String, path: String)] = [
        ("Claude Desktop",
         "~/Library/Application Support/Claude/claude_desktop_config.json"),
        ("Claude Code", "~/.claude.json"),
        ("Cursor", "~/.cursor/mcp.json"),
        ("Windsurf", "~/.codeium/windsurf/mcp_config.json"),
    ]

    static let searchDirs = [
        "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin",
        NSString(string: "~/.local/bin").expandingTildeInPath,
    ]

    static func findBinary(_ name: String) -> String? {
        for dir in searchDirs {
            let candidate = "\(dir)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        // Fall back to a login shell, which knows the user's real PATH.
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = ["-lc", "command -v \(name)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        guard (try? proc.run()) != nil else { return nil }
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else { return nil }
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                         encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (out?.isEmpty == false) ? out : nil
    }

    static func engineStatus() -> EngineStatus {
        EngineStatus(vaaraPath: findBinary("vaara"),
                     proxyPath: findBinary("vaara-mcp-proxy"))
    }

    private static func isGoverned(_ server: [String: Any]) -> Bool {
        (server["command"] as? String)?.hasSuffix("vaara-mcp-proxy") == true
    }

    static func scan() -> [MCPClient] {
        knownClients.compactMap { name, rawPath in
            let url = URL(fileURLWithPath: NSString(string: rawPath).expandingTildeInPath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                return MCPClient(id: name, configPath: url, governed: 0,
                                 ungoverned: 0, exists: false, hasBackup: false)
            }
            guard let data = try? Data(contentsOf: url),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let servers = obj["mcpServers"] as? [String: Any]
            else {
                return MCPClient(id: name, configPath: url, governed: 0,
                                 ungoverned: 0, exists: true, hasBackup: hasBackup(url))
            }
            var governed = 0, naked = 0
            for case let server as [String: Any] in servers.values {
                if isGoverned(server) { governed += 1 }
                else if server["command"] is String { naked += 1 }
                // URL-based (remote) servers are skipped in v1.
            }
            return MCPClient(id: name, configPath: url, governed: governed,
                             ungoverned: naked, exists: true, hasBackup: hasBackup(url))
        }
    }

    private static func backupURL(_ config: URL) -> URL {
        config.deletingLastPathComponent()
            .appendingPathComponent(config.lastPathComponent + ".vaara-backup")
    }

    static func hasBackup(_ config: URL) -> Bool {
        FileManager.default.fileExists(atPath: backupURL(config).path)
    }

    /// Rewrite every ungoverned stdio server through the proxy.
    /// Returns how many entries were rewritten, or nil on failure.
    /// The first backup is preserved forever (it is the pre-Vaara state);
    /// re-governing never overwrites it.
    @discardableResult
    static func govern(_ client: MCPClient, shadow: Bool = false) -> Int? {
        guard let proxy = findBinary("vaara-mcp-proxy") else { return nil }
        guard let data = try? Data(contentsOf: client.configPath),
              var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var servers = obj["mcpServers"] as? [String: Any]
        else { return nil }

        let backup = backupURL(client.configPath)
        if !FileManager.default.fileExists(atPath: backup.path) {
            try? data.write(to: backup)
        }

        let trailDB = NSString(string: "~/.vaara/mcp-proxy/audit.db").expandingTildeInPath
        try? FileManager.default.createDirectory(
            atPath: NSString(string: "~/.vaara/mcp-proxy").expandingTildeInPath,
            withIntermediateDirectories: true)

        var rewritten = 0
        for (name, value) in servers {
            guard var server = value as? [String: Any],
                  let command = server["command"] as? String,
                  !command.hasSuffix("vaara-mcp-proxy")
            else { continue }
            var args: [String] = ["--upstream", command]
            for arg in (server["args"] as? [String]) ?? [] {
                args += ["--upstream-arg", arg]
            }
            args += ["--db", trailDB, "--agent-id", "mcp:\(name)"]
            if shadow { args.append("--shadow") }
            server["command"] = proxy
            server["args"] = args
            servers[name] = server
            rewritten += 1
        }
        guard rewritten > 0 else { return 0 }
        obj["mcpServers"] = servers
        guard let out = try? JSONSerialization.data(
            withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        else { return nil }
        guard (try? out.write(to: client.configPath)) != nil else { return nil }
        return rewritten
    }

    /// Put the original (pre-Vaara) config back.
    static func restore(_ client: MCPClient) -> Bool {
        let backup = backupURL(client.configPath)
        guard let data = try? Data(contentsOf: backup) else { return false }
        return (try? data.write(to: client.configPath)) != nil
    }
}
