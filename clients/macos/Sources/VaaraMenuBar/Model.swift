// The gate model: watches one or more Vaara audit SQLite DBs and the
// approvals directory, ports core.py's logic 1:1 and aggregates.
//
//   green   latest decision in the window was allow (or nothing recent)
//   yellow  latest was escalate, or an approval is pending
//   red     latest was deny; ages back to green after the alert window

import AppKit
import Foundation
import SQLite3
import UserNotifications

enum GateState: String {
    case green, yellow, red

    var label: String {
        switch self {
        case .green:  return "Allowed"
        case .yellow: return "Escalated"
        case .red:    return "Denied"
        }
    }

    var detail: String {
        switch self {
        case .green:  return "The AI's moves are passing policy."
        case .yellow: return "A move needs, or got, human review."
        case .red:    return "The gate blocked the AI's latest move."
        }
    }
}

struct DecisionEvent: Identifiable {
    let id: String         // "<db path>#<seq>", unique across sources
    let seq: Int64
    let verdict: String    // "deny" | "escalate"
    let toolName: String
    let timestamp: Date
    let reason: String
    let riskScore: Double?
}

struct AgentSummary: Identifiable {
    let id: String          // agent_id
    let lastSeen: Date
    let state: GateState    // that agent's newest decision in the window
    let allowed: Int
    let escalated: Int
    let denied: Int
}

/// The four protection presets, thresholds mirrored from
/// `vaara mode list` (policy/modes.py). The plugin reads the chosen one
/// from ~/.vaara/claude-code/config.json key "protection".
struct Preset: Identifiable {
    let id: String
    let blurb: String
    let escalate: Double
    let deny: Double

    static let all: [Preset] = [
        Preset(id: "eco",         blurb: "Cuts loops short on borderline risk",
               escalate: 0.40, deny: 0.60),
        Preset(id: "balanced",    blurb: "Default operating point",
               escalate: 0.55, deny: 0.85),
        Preset(id: "performance", blurb: "Loose. For high-throughput pipelines",
               escalate: 0.70, deny: 0.92),
        Preset(id: "strict",      blurb: "Escalate on doubt. Lockdown windows",
               escalate: 0.30, deny: 0.55),
    ]
}

/// What a preset would have decided, replayed over the recent window.
struct PresetStats {
    var allowed = 0
    var escalated = 0
    var denied = 0
}

struct Config: Codable {
    var db_paths: [String] = [
        NSString(string: "~/.vaara/claude-code/audit.db").expandingTildeInPath
    ]
    var alert_window_minutes: Int = 5
    var notifications: Bool = true
    var appearance: String = "dark"       // dark | light
    var menubar_graph: Bool = true        // activity sparkline next to the mark
    /// Which decisions raise a system notification:
    /// "off" | "deny" (denials only) | "interventions" (denials + escalations)
    var notify_on: String = "interventions"
    /// Settings depth: "basic" | "professional" | "enterprise".
    /// Basic sees mode, notifications, updates; professional adds
    /// thresholds and tuning; enterprise adds multi-trail sources.
    var user_level: String = "basic"

    // Tolerate configs written before a field existed, including the
    // Python proto's single db_path key, which migrates into db_paths.
    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let base = Config()
        if let many = try c.decodeIfPresent([String].self, forKey: .db_paths), !many.isEmpty {
            db_paths = many
        } else if let one = try? c.decode(String.self, forKey: .legacy_db_path) {
            db_paths = [one]
        } else {
            db_paths = base.db_paths
        }
        alert_window_minutes = try c.decodeIfPresent(Int.self, forKey: .alert_window_minutes) ?? base.alert_window_minutes
        notifications = try c.decodeIfPresent(Bool.self, forKey: .notifications) ?? base.notifications
        appearance = try c.decodeIfPresent(String.self, forKey: .appearance) ?? base.appearance
        menubar_graph = try c.decodeIfPresent(Bool.self, forKey: .menubar_graph) ?? base.menubar_graph
        // Migrate the old bool: false means off, true keeps the default.
        if let mode = try c.decodeIfPresent(String.self, forKey: .notify_on) {
            notify_on = mode
        } else {
            notify_on = notifications ? base.notify_on : "off"
        }
        user_level = try c.decodeIfPresent(String.self, forKey: .user_level) ?? base.user_level
    }

    enum CodingKeys: String, CodingKey {
        case db_paths, alert_window_minutes, notifications, appearance, menubar_graph
        case notify_on, user_level
        case legacy_db_path = "db_path"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(db_paths, forKey: .db_paths)
        try c.encode(alert_window_minutes, forKey: .alert_window_minutes)
        try c.encode(notifications, forKey: .notifications)
        try c.encode(appearance, forKey: .appearance)
        try c.encode(menubar_graph, forKey: .menubar_graph)
        try c.encode(notify_on, forKey: .notify_on)
        try c.encode(user_level, forKey: .user_level)
        // Keep the Python proto readable from the same file.
        if let first = db_paths.first {
            try c.encode(first, forKey: .legacy_db_path)
        }
    }

    static var url: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".vaara/menubar.json")
    }

    static func load() -> Config {
        guard let data = try? Data(contentsOf: url),
              let cfg = try? JSONDecoder().decode(Config.self, from: data)
        else { return Config() }
        return cfg
    }

    func save() {
        try? FileManager.default.createDirectory(
            at: Self.url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(self) { try? data.write(to: Self.url) }
    }
}

@MainActor
final class GateModel: ObservableObject {
    @Published var state: GateState = .green
    @Published var feed: [DecisionEvent] = []          // live, since launch
    @Published var agents: [AgentSummary] = []
    @Published var config = Config.load() { didSet { config.save() } }
    @Published var activePreset: String = "balanced"
    @Published var enforcementMode: String = "protect"   // protect | watch
    /// Custom [escalate, deny] from plugin config.json, nil when a preset rules.
    @Published var customThresholds: [Double]?
    @Published var presetStats: [String: PresetStats] = [:]
    /// Menu-bar sparkline: last 12 buckets of 10 s, oldest first.
    /// Each bucket is (activity count, worst verdict in the bucket).
    @Published var buckets: [(Int, GateState)] = []

    /// How far back an agent still counts as "running" (seconds).
    static let runningWindow: Double = 15 * 60

    private var cursors: [String: Int64] = [:]
    private var handledApprovals = Set<String>()
    private var timer: Timer?
    private let approvalsDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".vaara/approvals")

    func start() {
        for path in config.db_paths { cursors[path] = maxSeq(path) }  // start at now
        UNUserNotificationCenter.current()
            .requestAuthorization(options: [.alert, .sound]) { _, _ in }
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.poll() }
        }
        poll()
    }

    func addSource(_ path: String) {
        guard !config.db_paths.contains(path) else { return }
        config.db_paths.append(path)
        cursors[path] = maxSeq(path)
    }

    /// Scan ~/.vaara for SQLite files that hold a Vaara trail
    /// (an audit_records table) and watch every one not already watched.
    /// Returns how many new trails were added. MCP proxies and older
    /// installs write their own DBs; this is how they get found.
    @discardableResult
    func discoverTrails() -> Int {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".vaara")
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]) else { return 0 }
        var added = 0
        for case let url as URL in walker {
            guard url.pathExtension == "db",
                  !config.db_paths.contains(url.path),
                  isVaaraTrail(url.path) else { continue }
            addSource(url.path)
            added += 1
        }
        return added
    }

    private func isVaaraTrail(_ path: String) -> Bool {
        withDB(path) { db in
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(
                db,
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'audit_records'",
                -1, &stmt, nil) == SQLITE_OK else { return false }
            return sqlite3_step(stmt) == SQLITE_ROW
        } ?? false
    }

    func removeSource(_ path: String) {
        config.db_paths.removeAll { $0 == path }
        cursors[path] = nil
    }

    private func poll() {
        for path in config.db_paths {
            let since = cursors[path] ?? maxSeq(path)
            for event in newDecisions(path, sinceSeq: since) {
                cursors[path] = max(cursors[path] ?? -1, event.seq)
                feed.insert(event, at: 0)
                notify(event)
            }
        }
        feed.sort { $0.timestamp > $1.timestamp }
        if feed.count > 30 { feed.removeLast(feed.count - 30) }
        handlePendingApprovals()
        state = overallState()
        agents = agentSummaries()
        activePreset = readPluginPreset()
        enforcementMode = readPluginMode()
        customThresholds = readPluginThresholds()
        presetStats = replayPresets()
        buckets = activityBuckets()
    }

    /// All-events activity over the last two minutes, in 12 buckets of
    /// 10 s, oldest first. Counts every audit record (the "AI is doing
    /// things" pulse); the color carries the worst decision per bucket.
    private func activityBuckets(count: Int = 12, width: Double = 10) -> [(Int, GateState)] {
        let now = Date().timeIntervalSince1970
        let start = now - Double(count) * width
        var counts = [Int](repeating: 0, count: count)
        var worst = [GateState](repeating: .green, count: count)

        for path in config.db_paths {
            withDB(path) { db -> Void in
                let sql = """
                    SELECT timestamp, event_type, data FROM audit_records
                    WHERE timestamp >= ?
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
                sqlite3_bind_double(stmt, 1, start)
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let ts = sqlite3_column_double(stmt, 0)
                    let idx = Int((ts - start) / width)
                    guard idx >= 0, idx < count else { continue }
                    counts[idx] += 1
                    let eventType = String(cString: sqlite3_column_text(stmt, 1))
                    var verdict: GateState = .green
                    if eventType == "action_blocked" {
                        verdict = .red
                    } else if eventType == "escalation_sent" {
                        verdict = .yellow
                    } else if eventType == "decision_made" {
                        switch (parseData(sqlite3_column_text(stmt, 2))["decision"] as? String) ?? "" {
                        case "deny":     verdict = .red
                        case "escalate": verdict = .yellow
                        default:         verdict = .green
                        }
                    }
                    if verdict == .red || (verdict == .yellow && worst[idx] == .green) {
                        worst[idx] = verdict
                    }
                }
            }
        }
        return zip(counts, worst).map { ($0, $1) }
    }

    private func readPluginThresholds() -> [Double]? {
        guard let data = try? Data(contentsOf: pluginConfigURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let t = obj["thresholds"] as? [String: Any],
              let escalate = t["escalate"] as? Double,
              let deny = t["deny"] as? Double,
              escalate >= 0, escalate <= deny, deny <= 1
        else { return nil }
        return [escalate, deny]
    }

    /// Write custom thresholds; the plugin applies them over the preset.
    func setCustomThresholds(escalate: Double, deny: Double) {
        let e = max(0, min(escalate, 1)), d = max(e, min(deny, 1))
        writePluginConfig(key: "thresholds",
                          value: ["escalate": e, "deny": d])
        customThresholds = [e, d]
    }

    /// Back to pure preset: drop the custom override.
    func clearCustomThresholds() {
        writePluginConfig(key: "thresholds", value: nil)
        customThresholds = nil
    }

    // MARK: - SQLite (read-only in intent, but WAL-aware)

    /// Open a watched trail for reading. The engine writes in WAL mode, and
    /// a pure SQLITE_OPEN_READONLY connection cannot read committed-but-
    /// uncheckpointed rows from the -wal sidecar (it needs to build the
    /// -shm shared index, which read-only forbids). So the newest events
    /// stay invisible until a checkpoint happens on the writer's side.
    ///
    /// We open read-write so SQLite can attach the WAL, then immediately set
    /// PRAGMA query_only = ON: the connection can never modify the trail, but
    /// it sees every committed row. If the file is genuinely not writable, we
    /// fall back to a plain read-only open (better a slightly stale view than
    /// none).
    private func withDB<T>(_ path: String, _ body: (OpaquePointer) -> T) -> T? {
        var db: OpaquePointer?
        var flags: Int32 = SQLITE_OPEN_READWRITE
        if sqlite3_open_v2(path, &db, flags, nil) != SQLITE_OK {
            if db != nil { sqlite3_close(db); db = nil }
            flags = SQLITE_OPEN_READONLY
            guard sqlite3_open_v2(path, &db, flags, nil) == SQLITE_OK else {
                if db != nil { sqlite3_close(db) }
                return nil
            }
        }
        guard let db else { return nil }
        defer { sqlite3_close(db) }
        if flags == SQLITE_OPEN_READWRITE {
            sqlite3_exec(db, "PRAGMA query_only = ON", nil, nil, nil)
        }
        sqlite3_busy_timeout(db, 2000)
        return body(db)
    }

    private func maxSeq(_ path: String) -> Int64 {
        withDB(path) { db in
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, "SELECT MAX(seq) FROM audit_records", -1, &stmt, nil) == SQLITE_OK,
                  sqlite3_step(stmt) == SQLITE_ROW else { return -1 }
            return sqlite3_column_int64(stmt, 0)
        } ?? -1
    }

    private func decisionEvent(
        _ path: String, _ stmt: OpaquePointer?, includeAllows: Bool = false
    ) -> DecisionEvent? {
        guard let stmt else { return nil }
        let eventType = String(cString: sqlite3_column_text(stmt, 1))
        let data = parseData(sqlite3_column_text(stmt, 4))
        let decision = (data["decision"] as? String) ?? ""
        let verdict: String
        if eventType == "action_blocked" || decision == "deny" {
            verdict = "deny"
        } else if eventType == "escalation_sent" || decision == "escalate" {
            verdict = "escalate"
        } else if includeAllows && eventType == "decision_made" && decision == "allow" {
            verdict = "allow"
        } else {
            return nil
        }
        let seq = sqlite3_column_int64(stmt, 0)
        return DecisionEvent(
            id: "\(path)#\(seq)",
            seq: seq,
            verdict: verdict,
            toolName: String(cString: sqlite3_column_text(stmt, 2)),
            timestamp: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 3)),
            reason: String((data["reason"] as? String ?? "").prefix(300)),
            riskScore: data["risk_score"] as? Double)
    }

    private let eventColumns =
        "SELECT seq, event_type, tool_name, timestamp, data FROM audit_records"

    private func newDecisions(_ path: String, sinceSeq: Int64) -> [DecisionEvent] {
        withDB(path) { db in
            let sql = eventColumns + """
                 WHERE seq > ? AND event_type IN
                  ('action_blocked', 'decision_made', 'escalation_sent')
                 ORDER BY seq
                """
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
            sqlite3_bind_int64(stmt, 1, sinceSeq)
            var events: [DecisionEvent] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let event = decisionEvent(path, stmt) { events.append(event) }
            }
            return events
        } ?? []
    }

    /// Full decision history across all sources, newest first: every
    /// allow, escalate, and deny ever recorded, independent of when the
    /// app was launched. The rolling register.
    func history(limit: Int = 100) -> [DecisionEvent] {
        var all: [DecisionEvent] = []
        for path in config.db_paths {
            let events: [DecisionEvent] = withDB(path) { db in
                let sql = eventColumns + """
                     WHERE event_type IN
                      ('action_blocked', 'decision_made', 'escalation_sent')
                     ORDER BY seq DESC LIMIT 1000
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
                var out: [DecisionEvent] = []
                while sqlite3_step(stmt) == SQLITE_ROW, out.count < limit {
                    if let event = decisionEvent(path, stmt, includeAllows: true) {
                        out.append(event)
                    }
                }
                return out
            } ?? []
            all.append(contentsOf: events)
        }
        return Array(all.sorted { $0.timestamp > $1.timestamp }.prefix(limit))
    }

    /// Recent interventions (deny/escalate) for one agent, newest first.
    func agentInterventions(_ agentID: String, limit: Int = 20) -> [DecisionEvent] {
        var all: [DecisionEvent] = []
        for path in config.db_paths {
            let events: [DecisionEvent] = withDB(path) { db in
                let sql = eventColumns + """
                     WHERE agent_id = ? AND event_type IN
                      ('action_blocked', 'decision_made', 'escalation_sent')
                     ORDER BY seq DESC LIMIT 200
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
                sqlite3_bind_text(stmt, 1, agentID, -1,
                                  unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                var out: [DecisionEvent] = []
                while sqlite3_step(stmt) == SQLITE_ROW, out.count < limit {
                    if let event = decisionEvent(path, stmt) { out.append(event) }
                }
                return out
            } ?? []
            all.append(contentsOf: events)
        }
        return Array(all.sorted { $0.timestamp > $1.timestamp }.prefix(limit))
    }

    // MARK: - State (newest decision across all sources wins)

    private func overallState() -> GateState {
        if !pendingApprovals().isEmpty { return .yellow }
        let cutoff = Date().timeIntervalSince1970
            - Double(config.alert_window_minutes) * 60

        var newest: (ts: Double, state: GateState)?
        for path in config.db_paths {
            let found: (Double, GateState)? = withDB(path) { db in
                let sql = """
                    SELECT event_type, data, timestamp FROM audit_records
                    WHERE timestamp >= ? AND event_type IN
                      ('action_blocked', 'escalation_sent', 'decision_made')
                    ORDER BY seq DESC
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
                sqlite3_bind_double(stmt, 1, cutoff)
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let eventType = String(cString: sqlite3_column_text(stmt, 0))
                    let ts = sqlite3_column_double(stmt, 2)
                    if eventType == "action_blocked" { return (ts, .red) }
                    if eventType == "escalation_sent" { return (ts, .yellow) }
                    switch (parseData(sqlite3_column_text(stmt, 1))["decision"] as? String) ?? "" {
                    case "deny":     return (ts, .red)
                    case "escalate": return (ts, .yellow)
                    case "allow":    return (ts, .green)
                    default:         continue
                    }
                }
                return nil
            } ?? nil
            if let found, found.0 > (newest?.ts ?? -1) {
                newest = (found.0, found.1)
            }
        }
        return newest?.state ?? .green
    }

    /// Agents with any audit record inside runningWindow, newest first,
    /// merged across sources by agent_id.
    private func agentSummaries() -> [AgentSummary] {
        let cutoff = Date().timeIntervalSince1970 - Self.runningWindow
        struct Acc { var lastSeen = Date.distantPast
                     var state: GateState?
                     var stateTS = Date.distantPast
                     var allowed = 0; var escalated = 0; var denied = 0 }
        var acc: [String: Acc] = [:]

        for path in config.db_paths {
            withDB(path) { db -> Void in
                let sql = """
                    SELECT agent_id, event_type, data, timestamp
                    FROM audit_records
                    WHERE timestamp >= ?
                    ORDER BY seq DESC
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
                sqlite3_bind_double(stmt, 1, cutoff)

                while sqlite3_step(stmt) == SQLITE_ROW {
                    let agent = String(cString: sqlite3_column_text(stmt, 0))
                    let eventType = String(cString: sqlite3_column_text(stmt, 1))
                    let ts = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 3))
                    if acc[agent] == nil { acc[agent] = Acc() }
                    acc[agent]!.lastSeen = max(acc[agent]!.lastSeen, ts)

                    let verdict: GateState?
                    switch eventType {
                    case "action_blocked":  verdict = .red
                    case "escalation_sent": verdict = .yellow
                    case "decision_made":
                        switch (parseData(sqlite3_column_text(stmt, 2))["decision"] as? String) ?? "" {
                        case "deny":     verdict = .red
                        case "escalate": verdict = .yellow
                        case "allow":    verdict = .green
                        default:         verdict = nil
                        }
                    default: verdict = nil
                    }
                    guard let verdict else { continue }
                    if ts > acc[agent]!.stateTS {
                        acc[agent]!.state = verdict
                        acc[agent]!.stateTS = ts
                    }
                    switch verdict {
                    case .green:  acc[agent]!.allowed += 1
                    case .yellow: acc[agent]!.escalated += 1
                    case .red:    acc[agent]!.denied += 1
                    }
                }
            }
        }
        return acc
            .map { id, a in
                AgentSummary(id: id, lastSeen: a.lastSeen,
                             state: a.state ?? .green,
                             allowed: a.allowed, escalated: a.escalated,
                             denied: a.denied)
            }
            .sorted { $0.lastSeen > $1.lastSeen }
    }

    private func parseData(_ text: UnsafePointer<UInt8>?) -> [String: Any] {
        guard let text,
              let data = String(cString: text).data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return obj
    }

    // MARK: - Protection presets (the plugin's threshold profile)

    private let pluginConfigURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".vaara/claude-code/config.json")

    private func readPluginPreset() -> String {
        guard let data = try? Data(contentsOf: pluginConfigURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let preset = obj["protection"] as? String, !preset.isEmpty
        else { return "balanced" }
        return preset
    }

    private func readPluginMode() -> String {
        guard let data = try? Data(contentsOf: pluginConfigURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mode = obj["mode"] as? String, !mode.isEmpty
        else { return "protect" }
        return mode
    }

    /// Set the plugin's protection preset (and drop any custom override,
    /// so the click means what it says).
    func setPreset(_ id: String) {
        writePluginConfig(key: "protection", value: id)
        writePluginConfig(key: "thresholds", value: nil)
        activePreset = id
        customThresholds = nil
    }

    /// Flip the plugin between blocking (protect) and shadow (watch).
    func setMode(_ mode: String) {
        writePluginConfig(key: "mode", value: mode)
        enforcementMode = mode
    }

    /// Merge one key into config.json, preserving every other key
    /// (/vaara-setup owns the file's shape). nil removes the key.
    private func writePluginConfig(key: String, value: Any?) {
        var obj: [String: Any] = [:]
        if let data = try? Data(contentsOf: pluginConfigURL),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            obj = existing
        }
        if let value { obj[key] = value } else { obj.removeValue(forKey: key) }
        try? FileManager.default.createDirectory(
            at: pluginConfigURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(
            withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: pluginConfigURL)
        }
    }

    /// Replay the recent window's risk scores against every preset's
    /// thresholds: real data on what each one would have decided.
    private func replayPresets() -> [String: PresetStats] {
        let cutoff = Date().timeIntervalSince1970 - Self.runningWindow
        var scores: [Double] = []
        for path in config.db_paths {
            let found: [Double] = withDB(path) { db in
                let sql = """
                    SELECT data FROM audit_records
                    WHERE timestamp >= ? AND event_type IN
                      ('action_blocked', 'decision_made')
                    """
                var stmt: OpaquePointer?
                defer { sqlite3_finalize(stmt) }
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
                sqlite3_bind_double(stmt, 1, cutoff)
                var out: [Double] = []
                while sqlite3_step(stmt) == SQLITE_ROW {
                    if let risk = parseData(sqlite3_column_text(stmt, 0))["risk_score"] as? Double {
                        out.append(risk)
                    }
                }
                return out
            } ?? []
            scores.append(contentsOf: found)
        }

        var stats: [String: PresetStats] = [:]
        var lenses: [(String, Double, Double)] = Preset.all.map { ($0.id, $0.escalate, $0.deny) }
        if let custom = customThresholds, custom.count == 2 {
            lenses.append(("custom", custom[0], custom[1]))
        }
        for (id, escalate, deny) in lenses {
            var s = PresetStats()
            for risk in scores {
                if risk >= deny { s.denied += 1 }
                else if risk >= escalate { s.escalated += 1 }
                else { s.allowed += 1 }
            }
            stats[id] = s
        }
        return stats
    }

    // MARK: - Approval handshake (file-based, shared with the Python proto)

    private func pendingApprovals() -> [[String: Any]] {
        guard let names = try? FileManager.default
            .contentsOfDirectory(atPath: approvalsDir.path) else { return [] }
        var out: [[String: Any]] = []
        for name in names.sorted() where name.hasSuffix(".request.json") {
            let url = approvalsDir.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url),
                  let req = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            let requestedAt = (req["requested_at"] as? Double) ?? 0
            if Date().timeIntervalSince1970 - requestedAt > 600 { continue } // stale
            out.append(req)
        }
        return out
    }

    private func handlePendingApprovals() {
        for req in pendingApprovals() {
            guard let actionID = req["action_id"] as? String,
                  !actionID.isEmpty, !handledApprovals.contains(actionID) else { continue }
            handledApprovals.insert(actionID)

            let alert = NSAlert()
            alert.messageText = "Vaara: approval needed"
            alert.informativeText =
                "\(req["tool_name"] as? String ?? "?")\n\n\(req["reason"] as? String ?? "")"
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Deny")
            alert.addButton(withTitle: "Approve")
            NSApp.activate(ignoringOtherApps: true)
            let decision = alert.runModal() == .alertSecondButtonReturn ? "approve" : "deny"

            let payload: [String: Any] =
                ["decision": decision, "decided_at": Date().timeIntervalSince1970]
            if let data = try? JSONSerialization.data(withJSONObject: payload) {
                try? data.write(to: approvalsDir
                    .appendingPathComponent("\(actionID).decision.json"))
            }
        }
    }

    // MARK: - Update check (GitHub latest release, on demand only)

    @Published var updateStatus: String?

    /// Compare the installed engine's version against the latest GitHub
    /// release tag. Runs only when the user clicks; no background phoning.
    func checkForUpdates() {
        updateStatus = "checking..."
        let installed = installedEngineVersion()
        var req = URLRequest(
            url: URL(string: "https://api.github.com/repos/vaaraio/vaara/releases/latest")!)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let latest: String? = data
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
                .flatMap { $0["tag_name"] as? String }
                .map { $0.hasPrefix("v") ? String($0.dropFirst()) : $0 }
            Task { @MainActor in
                guard let latest else {
                    self.updateStatus = "could not reach github.com"
                    return
                }
                if let installed, installed == latest {
                    self.updateStatus = "up to date (\(installed))"
                } else if let installed {
                    self.updateStatus =
                        "\(latest) available (installed \(installed)). "
                        + "Update: brew upgrade vaara"
                } else {
                    self.updateStatus = "latest is \(latest); engine not found"
                }
            }
        }.resume()
    }

    private func installedEngineVersion() -> String? {
        guard let vaara = SetupScanner.engineStatus().vaaraPath else { return nil }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: vaara)
        proc.arguments = ["--version"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        guard (try? proc.run()) != nil else { return nil }
        proc.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                         encoding: .utf8) ?? ""
        // "vaara 1.40.0" or bare "1.40.0"
        let token = out.split(separator: " ").last.map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (token?.isEmpty ?? true) ? nil : token
    }

    // MARK: - Notifications

    private func notify(_ event: DecisionEvent) {
        switch config.notify_on {
        case "off":  return
        case "deny": guard event.verdict == "deny" else { return }
        default:     break  // "interventions": deny + escalate (the feed)
        }
        guard config.notifications else { return }
        let content = UNMutableNotificationContent()
        content.title = "Vaara: \(event.verdict.uppercased())"
        content.subtitle = event.toolName
        content.body = event.reason.isEmpty ? "no reason recorded" : event.reason
        content.sound = .default
        UNUserNotificationCenter.current().add(
            UNNotificationRequest(identifier: UUID().uuidString,
                                  content: content, trigger: nil))
    }
}
