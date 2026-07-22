// The popover. Aesthetic brief: Roon on glass — translucent material,
// unhurried spacing, quiet typography, one accent color at a time.
// Appearance is a single click: sun/moon in the footer.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

extension GateState {
    var color: Color {
        switch self {
        case .green:  return Color(red: 0.37, green: 0.72, blue: 0.47)
        case .yellow: return Color(red: 0.90, green: 0.76, blue: 0.29)
        case .red:    return Color(red: 0.85, green: 0.34, blue: 0.34)
        }
    }
}

struct Palette {
    let ink: Color, faint: Color, ghost: Color, hairline: Color, wash: Color

    // Dark mirrors the webpage: --bg #0F1417, --ink #DEE4E1,
    // --muted #8B9792, --faint #5E6A66, --line sage at 0.16.
    static let dark = Palette(
        ink: Color(red: 0.871, green: 0.894, blue: 0.882),
        faint: Color(red: 0.545, green: 0.592, blue: 0.573),
        ghost: Color(red: 0.369, green: 0.416, blue: 0.400),
        hairline: Color(red: 0.482, green: 0.612, blue: 0.541).opacity(0.16),
        wash: Color(red: 0.059, green: 0.078, blue: 0.090).opacity(0.85))
    // Light is pure white, fully opaque: the glass may live at the
    // window edge, but the surface is paper.
    static let light = Palette(
        ink: .black.opacity(0.95), faint: .black.opacity(0.62),
        ghost: .black.opacity(0.42), hairline: .black.opacity(0.12),
        wash: Color.white)
}

/// Bump on every source change; shown in the footer so a stale build is
/// visible at a glance instead of masquerading as a bug.
let BUILD_STAMP = "b35 · 2026-07-22"

struct ContentView: View {
    @ObservedObject var model: GateModel
    @State private var screen: Screen = .overview
    @State private var selectedAgent: AgentSummary?
    @State private var discoveredCount: Int?

    enum Screen { case overview, settings, history, setup }
    @State private var engine = SetupScanner.engineStatus()
    @State private var clients: [MCPClient] = SetupScanner.scan()
    @State private var installing = false
    @State private var installLog: String?

    private var dark: Bool { model.config.appearance != "light" }
    private var p: Palette { dark ? .dark : .light }

    var body: some View {
        Group {
            if let pending = model.pendingApproval {
                approvalPanel(pending)
            } else {
                mainBody
            }
        }
        .frame(width: 400)
        .background(p.wash)
        .background(.ultraThinMaterial)
        .environment(\.colorScheme, dark ? .dark : .light)
    }

    private var mainBody: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let agent = selectedAgent {
                agentDetail(agent)
            } else {
                header
                Rectangle().fill(p.hairline).frame(height: 1)
                switch screen {
                case .overview: overview
                case .settings: settings
                case .history:  historyView
                case .setup:    setupView
                }
            }
            Rectangle().fill(p.hairline).frame(height: 1)
            footer
        }
    }

    // MARK: approval panel — the gate, made human. Shown over everything
    // while an escalated action waits. The blocked hook is polling for the
    // decision file; Block is the safe default, Approve is the only way
    // through, and doing nothing lets the hook time out fail-closed.

    private func approvalPanel(_ pending: PendingApproval) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Circle()
                    .fill(GateState.yellow.color)
                    .frame(width: 11, height: 11)
                    .shadow(color: GateState.yellow.color.opacity(0.6), radius: 7)
                Text("Approval needed")
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(p.ink)
                Spacer()
            }
            .padding(.horizontal, 22)
            .padding(.top, 24)
            .padding(.bottom, 16)

            Rectangle().fill(p.hairline).frame(height: 1)

            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("AN AI AGENT WANTS TO")
                        .font(.system(size: 9, weight: .medium))
                        .tracking(1.2)
                        .foregroundStyle(p.ghost)
                    Text(pending.toolName)
                        .font(.system(size: 15, design: .monospaced))
                        .foregroundStyle(p.ink)
                        .lineLimit(2).truncationMode(.middle)
                }
                if !pending.reason.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("WHY IT WAS FLAGGED")
                            .font(.system(size: 9, weight: .medium))
                            .tracking(1.2)
                            .foregroundStyle(p.ghost)
                        Text(pending.reason)
                            .font(.system(size: 12))
                            .foregroundStyle(p.faint)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 18)

            Rectangle().fill(p.hairline).frame(height: 1)

            HStack(spacing: 12) {
                Button {
                    model.resolveApproval(pending.actionID, approve: false)
                } label: {
                    Text("Block")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(GateState.red.color)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)

                Button {
                    model.resolveApproval(pending.actionID, approve: true)
                } label: {
                    Text("Approve")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(dark ? .black : .white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(GateState.green.color)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 16)

            Text("Doing nothing blocks the action when the request times out.")
                .font(.system(size: 10))
                .foregroundStyle(p.ghost)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.bottom, 16)
        }
    }

    // MARK: history — the full shadow register, launch-independent

    private var historyView: some View {
        let events = model.history()
        return Group {
            if events.isEmpty {
                VStack(spacing: 10) {
                    Text("The watched trails hold no interventions.")
                        .font(.system(size: 12))
                        .foregroundStyle(p.ghost)
                    Text("Other tools (an MCP proxy, an older install)\nkeep their own trails.")
                        .font(.system(size: 11))
                        .foregroundStyle(p.ghost)
                        .multilineTextAlignment(.center)
                    Button("Find trails in ~/.vaara") {
                        _ = model.discoverTrails()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(p.faint)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 36)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        sectionLabel("EVERY DECISION ON RECORD")
                        ForEach(events) { HistoryRow(event: $0, p: p) }
                    }
                    .padding(.vertical, 8)
                }
                .frame(maxHeight: 560)
            }
        }
    }

    // MARK: header — the state, stated once, calmly

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            Circle()
                .fill(model.state.color)
                .frame(width: 10, height: 10)
                .shadow(color: model.state.color.opacity(0.55), radius: 6)
                .offset(y: -1)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.state.label)
                    .font(.system(size: 24, weight: .light))
                    .foregroundStyle(p.ink)
                Text(model.state.detail)
                    .font(.system(size: 12))
                    .foregroundStyle(p.faint)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 22)
        .padding(.bottom, 18)
        .animation(.easeInOut(duration: 0.25), value: model.state)
    }

    // MARK: overview — running agents, then the intervention feed

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                if !model.agents.isEmpty {
                    sectionLabel("RUNNING NOW")
                    ForEach(model.agents) { agent in
                        AgentRow(agent: agent, p: p)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                withAnimation(.easeInOut(duration: 0.15)) {
                                    selectedAgent = agent
                                }
                            }
                    }
                    Rectangle().fill(p.hairline).frame(height: 1)
                        .padding(.vertical, 6)
                }
                if model.feed.isEmpty && model.agents.isEmpty && model.history().isEmpty {
                    VStack(spacing: 12) {
                        Text("Nothing is governed yet.")
                            .font(.system(size: 13))
                            .foregroundStyle(p.faint)
                        Text("Open Setup, point your AI tools through\nVaara, and their moves show up here.")
                            .font(.system(size: 11.5))
                            .foregroundStyle(p.ghost)
                            .multilineTextAlignment(.center)
                        Button("Open Setup") {
                            withAnimation(.easeInOut(duration: 0.15)) { screen = .setup }
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(GateState.green.color)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 32)
                } else if model.feed.isEmpty {
                    Text("No interventions yet.\nAllowed moves pass in silence.")
                        .font(.system(size: 12))
                        .foregroundStyle(p.ghost)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 36)
                } else {
                    sectionLabel("INTERVENTIONS")
                    ForEach(model.feed) { FeedRow(event: $0, p: p) }
                }
            }
            .padding(.vertical, 8)
        }
        .frame(maxHeight: 560)
    }

    // MARK: agent detail — who behaved

    private func agentDetail(_ agent: AgentSummary) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { selectedAgent = nil }
                } label: {
                    Image(systemName: "chevron.left").font(.system(size: 12, weight: .medium))
                }
                .buttonStyle(.plain)
                .foregroundStyle(p.faint)
                Circle().fill(agent.state.color).frame(width: 8, height: 8)
                    .shadow(color: agent.state.color.opacity(0.55), radius: 5)
                Text(agent.id)
                    .font(.system(size: 16, weight: .light))
                    .foregroundStyle(p.ink)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 14)

            HStack(spacing: 18) {
                stat(agent.allowed, "allowed", GateState.green.color)
                stat(agent.escalated, "escalated", GateState.yellow.color)
                stat(agent.denied, "denied", GateState.red.color)
                Spacer()
                Text(agent.lastSeen, format: .dateTime.hour().minute())
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(p.ghost)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 16)

            Rectangle().fill(p.hairline).frame(height: 1)

            let events = model.agentInterventions(agent.id)
            if events.isEmpty {
                Text("Nothing was flagged for this agent.")
                    .font(.system(size: 12))
                    .foregroundStyle(p.ghost)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 36)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(events) { FeedRow(event: $0, p: p) }
                    }
                    .padding(.vertical, 6)
                }
                .frame(maxHeight: 420)
            }
        }
    }

    private func stat(_ n: Int, _ label: String, _ color: Color) -> some View {
        HStack(spacing: 5) {
            Text("\(n)")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundStyle(n > 0 ? color : p.ghost)
            Text(label).font(.system(size: 11)).foregroundStyle(p.faint)
        }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
            .tracking(1.2)
            .foregroundStyle(p.ghost)
            .padding(.horizontal, 20)
            .padding(.top, 8)
            .padding(.bottom, 4)
    }

    // MARK: setup — the app wires the AIs in itself

    private var setupView: some View {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    sectionLabelPlain("ENGINE")
                    HStack(spacing: 8) {
                        Circle()
                            .fill(engine.ok ? GateState.green.color : GateState.red.color)
                            .frame(width: 7, height: 7)
                        Text(engine.ok
                             ? "vaara found: \(engine.vaaraPath ?? "")"
                             : "vaara engine not installed on this Mac")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(p.faint)
                            .lineLimit(1).truncationMode(.middle)
                    }
                    if !engine.ok {
                        HStack(spacing: 10) {
                            Button(installing ? "Installing..." : "Install engine") {
                                installEngine()
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(installing ? p.ghost : GateState.green.color)
                            .disabled(installing)
                            Text("one-time, via Homebrew")
                                .font(.system(size: 10))
                                .foregroundStyle(p.ghost)
                        }
                        if let log = installLog {
                            Text(log)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(p.ghost)
                                .lineLimit(2).truncationMode(.middle)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    sectionLabelPlain("AI CLIENTS ON THIS MAC")
                    ForEach(clients.filter(\.exists)) { client in
                        clientRow(client)
                    }
                    if clients.filter(\.exists).isEmpty {
                        Text("No MCP client configs found.")
                            .font(.system(size: 11))
                            .foregroundStyle(p.ghost)
                    }
                }

                Text("Govern rewrites the client's MCP servers to run through "
                     + "vaara-mcp-proxy. The original config is backed up first "
                     + "and Restore puts it back. Restart the client to apply.")
                    .font(.system(size: 10))
                    .foregroundStyle(p.ghost)

                Button("Rescan") {
                    engine = SetupScanner.engineStatus()
                    clients = SetupScanner.scan()
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundStyle(p.faint)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func clientRow(_ client: MCPClient) -> some View {
        HStack(spacing: 10) {
            Circle()
                .fill(client.ungoverned == 0
                      ? GateState.green.color
                      : GateState.yellow.color)
                .frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 1) {
                Text(client.id).font(.system(size: 12.5)).foregroundStyle(p.ink)
                Text(client.ungoverned == 0
                     ? (client.governed > 0
                        ? "\(client.governed) server\(client.governed == 1 ? "" : "s") governed"
                        : "no MCP servers configured")
                     : "\(client.ungoverned) ungoverned server\(client.ungoverned == 1 ? "" : "s")"
                       + (client.governed > 0 ? ", \(client.governed) governed" : ""))
                    .font(.system(size: 10.5))
                    .foregroundStyle(p.ghost)
            }
            Spacer()
            if client.ungoverned > 0 && engine.ok {
                Button("Govern") {
                    _ = SetupScanner.govern(client)
                    clients = SetupScanner.scan()
                }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(GateState.green.color)
            }
            if client.hasBackup {
                Button("Restore") {
                    _ = SetupScanner.restore(client)
                    clients = SetupScanner.scan()
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundStyle(p.ghost)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: settings, inline — no second window, nothing modal

    private var pro: Bool { model.config.user_level != "basic" }
    private var enterprise: Bool { model.config.user_level == "enterprise" }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("SETTINGS FOR")
                Picker("", selection: $model.config.user_level) {
                    Text("Basic").tag("basic")
                    Text("Professional").tag("professional")
                    Text("Enterprise").tag("enterprise")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text(model.config.user_level == "basic"
                     ? "The essentials. Everything else keeps its defaults."
                     : model.config.user_level == "professional"
                     ? "Adds thresholds and tuning."
                     : "Adds multiple trails and every control.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(p.ghost)
            }

            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("GATE")
                Picker("", selection: Binding(
                    get: { model.enforcementMode },
                    set: { model.setMode($0) })) {
                    Text("Block").tag("protect")
                    Text("Shadow").tag("watch")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Text(model.enforcementMode == "watch"
                     ? "Recording only. Nothing gets blocked."
                     : "Enforcing. Deny decisions stop the call.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(p.ghost)
            }

            if pro {
                VStack(alignment: .leading, spacing: 6) {
                    sectionLabelPlain("PROTECTION · WOULD HAVE DECIDED (15 MIN)")
                    ForEach(Preset.all) { preset in
                        presetRow(preset)
                    }
                    customRow
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("NOTIFY ON")
                Picker("", selection: $model.config.notify_on) {
                    Text("Nothing").tag("off")
                    Text("Denials").tag("deny")
                    Text("All interventions").tag("interventions")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            if pro {
                Toggle(isOn: $model.config.menubar_graph) {
                    Text("Activity graph in the menu bar")
                        .font(.system(size: 13)).foregroundStyle(p.ink)
                }
                .toggleStyle(.switch)
                .controlSize(.mini)
                .tint(model.state.color)

                VStack(alignment: .leading, spacing: 8) {
                    sectionLabelPlain("SIGNAL FADES AFTER")
                    Picker("", selection: $model.config.alert_window_minutes) {
                        ForEach([1, 5, 15, 60], id: \.self) { Text("\($0) min").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("UPDATES")
                HStack(spacing: 12) {
                    Button("Check for updates") { model.checkForUpdates() }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(p.ink.opacity(0.7))
                    if let status = model.updateStatus {
                        Text(status)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(p.ghost)
                            .lineLimit(2)
                    }
                }
                Text("app build \(BUILD_STAMP)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(p.ghost)
            }

            if enterprise {
            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("WATCHING")
                ForEach(model.config.db_paths, id: \.self) { path in
                    HStack {
                        Text(path)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(p.faint)
                            .lineLimit(1)
                            .truncationMode(.head)
                        Spacer()
                        if model.config.db_paths.count > 1 {
                            Button {
                                model.removeSource(path)
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.system(size: 8, weight: .medium))
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(p.ghost)
                        }
                    }
                }
                HStack(spacing: 14) {
                    Button("Add a trail...") { chooseDB() }
                    Button("Find trails in ~/.vaara") {
                        discoveredCount = model.discoverTrails()
                    }
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundStyle(p.ink.opacity(0.7))
                if let n = discoveredCount {
                    Text(n == 0 ? "No new trails found."
                                : "Added \(n) trail\(n == 1 ? "" : "s").")
                        .font(.system(size: 10))
                        .foregroundStyle(p.ghost)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                sectionLabelPlain("APPROVALS FOLDER")
                Text(model.config.approvals_dir ?? "~/.vaara/approvals (default)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(p.faint)
                    .lineLimit(1).truncationMode(.head)
                HStack(spacing: 14) {
                    Button("Choose folder...") { chooseApprovalsDir() }
                    if model.config.approvals_dir != nil {
                        Button("Reset to default") {
                            model.config.approvals_dir = nil
                        }
                    }
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundStyle(p.ink.opacity(0.7))
                Text("Where the app watches for escalations needing your "
                     + "approval. Point it at the engine's approvals folder "
                     + "for bridge or multi-machine setups.")
                    .font(.system(size: 9))
                    .foregroundStyle(p.ghost)
            }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var customRow: some View {
        let active = model.customThresholds != nil
        let escalate = model.customThresholds?.first ?? 0.55
        let deny = model.customThresholds?.last ?? 0.85
        let stats = model.presetStats["custom"]
        return HStack(spacing: 10) {
            Button {
                if active {
                    model.clearCustomThresholds()
                } else {
                    model.setCustomThresholds(escalate: escalate, deny: deny)
                }
            } label: {
                HStack(spacing: 10) {
                    Circle()
                        .strokeBorder(active ? model.state.color : p.ghost, lineWidth: 1)
                        .background(Circle().fill(active ? model.state.color : .clear))
                        .frame(width: 7, height: 7)
                    Text("Custom")
                        .font(.system(size: 12, weight: active ? .medium : .regular))
                        .foregroundStyle(active ? p.ink : p.faint)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Spacer()

            Stepper(String(format: "esc %.2f", escalate),
                    value: Binding(
                        get: { escalate },
                        set: { model.setCustomThresholds(escalate: $0, deny: max($0, deny)) }),
                    in: 0...1, step: 0.05)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(active ? p.faint : p.ghost)
                .controlSize(.mini)
            Stepper(String(format: "deny %.2f", deny),
                    value: Binding(
                        get: { deny },
                        set: { model.setCustomThresholds(escalate: min(escalate, $0), deny: $0) }),
                    in: 0...1, step: 0.05)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(active ? p.faint : p.ghost)
                .controlSize(.mini)

            if active, let stats {
                HStack(spacing: 7) {
                    Text("\(stats.allowed)")
                        .foregroundStyle(GateState.green.color.opacity(0.8))
                    Text("\(stats.escalated)")
                        .foregroundStyle(GateState.yellow.color.opacity(0.8))
                    Text("\(stats.denied)")
                        .foregroundStyle(GateState.red.color.opacity(0.8))
                }
                .font(.system(size: 11, design: .monospaced))
            }
        }
        .padding(.vertical, 4)
    }

    private func presetRow(_ preset: Preset) -> some View {
        let active = model.activePreset == preset.id && model.customThresholds == nil
        let stats = model.presetStats[preset.id] ?? PresetStats()
        return Button {
            model.setPreset(preset.id)
        } label: {
            HStack(spacing: 10) {
                Circle()
                    .strokeBorder(active ? model.state.color : p.ghost, lineWidth: 1)
                    .background(Circle().fill(active ? model.state.color : .clear))
                    .frame(width: 7, height: 7)
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(preset.id.capitalized)
                            .font(.system(size: 12, weight: active ? .medium : .regular))
                            .foregroundStyle(active ? p.ink : p.faint)
                        Text(String(format: "esc %.2f · deny %.2f",
                                    preset.escalate, preset.deny))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(p.ghost)
                    }
                    Text(preset.blurb)
                        .font(.system(size: 10))
                        .foregroundStyle(p.ghost)
                }
                Spacer()
                HStack(spacing: 7) {
                    Text("\(stats.allowed)")
                        .foregroundStyle(GateState.green.color.opacity(0.8))
                    Text("\(stats.escalated)")
                        .foregroundStyle(GateState.yellow.color.opacity(0.8))
                    Text("\(stats.denied)")
                        .foregroundStyle(GateState.red.color.opacity(0.8))
                }
                .font(.system(size: 11, design: .monospaced))
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func sectionLabelPlain(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
            .tracking(1.2)
            .foregroundStyle(p.ghost)
    }

    private func chooseApprovalsDir() {
        let panel = NSOpenPanel()
        panel.message = "Pick the folder the engine writes approval requests to"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.showsHiddenFiles = true
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            model.config.approvals_dir = url.path
        }
    }

    private func chooseDB() {
        let panel = NSOpenPanel()
        panel.message = "Pick the Vaara audit DB to watch"
        panel.allowedContentTypes = [UTType(filenameExtension: "db") ?? .data]
        panel.canChooseDirectories = false
        panel.showsHiddenFiles = true
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let url = panel.url {
            model.addSource(url.path)
        }
    }

    private func installEngine() {
        installing = true
        installLog = "starting Homebrew..."
        SetupScanner.installEngine(
            progress: { line in installLog = line },
            done: { ok in
                installing = false
                engine = SetupScanner.engineStatus()
                clients = SetupScanner.scan()
                installLog = ok && engine.ok
                    ? "engine installed: \(engine.vaaraPath ?? "")"
                    : (ok ? "install finished; engine still not on PATH"
                          : installLog)
            })
    }

    // MARK: footer

    private var footer: some View {
        HStack(spacing: 16) {
            footerTab("Now", .overview)
            footerTab("History", .history)
            footerTab("Setup", .setup)
            footerTab("Settings", .settings)
            Button {
                model.config.appearance = dark ? "light" : "dark"
            } label: {
                Image(systemName: dark ? "sun.max" : "moon")
                    .font(.system(size: 12))
            }
            .help(dark ? "Switch to light" : "Switch to dark")
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
        }
        .buttonStyle(.plain)
        .font(.system(size: 12))
        .foregroundStyle(p.faint)
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private func footerTab(_ label: String, _ target: Screen) -> some View {
        Button(label) {
            withAnimation(.easeInOut(duration: 0.15)) {
                selectedAgent = nil
                screen = target
            }
        }
        .foregroundStyle(screen == target && selectedAgent == nil ? p.ink : p.faint)
    }
}

private struct AgentRow: View {
    let agent: AgentSummary
    let p: Palette

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(agent.state.color)
                .frame(width: 7, height: 7)
                .shadow(color: agent.state.color.opacity(0.5), radius: 4)
            Text(agent.id)
                .font(.system(size: 12.5))
                .foregroundStyle(p.ink)
                .lineLimit(1).truncationMode(.middle)
            Spacer()
            HStack(spacing: 8) {
                if agent.denied > 0 {
                    Text("\(agent.denied)").font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(GateState.red.color)
                }
                if agent.escalated > 0 {
                    Text("\(agent.escalated)").font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(GateState.yellow.color)
                }
                Text("\(agent.allowed)").font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(p.ghost)
            }
            Image(systemName: "chevron.right")
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(p.ghost)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 7)
    }
}

private struct HistoryRow: View {
    let event: DecisionEvent
    let p: Palette

    private var color: Color {
        switch event.verdict {
        case "deny":     return GateState.red.color
        case "escalate": return GateState.yellow.color
        default:         return GateState.green.color
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            Text(event.timestamp, format: .dateTime.day().month().hour().minute())
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(p.ghost)
                .frame(width: 78, alignment: .leading)
            Rectangle().fill(color).frame(width: 2, height: 22).cornerRadius(1)
            VStack(alignment: .leading, spacing: 1) {
                Text(event.toolName)
                    .font(.system(size: 12))
                    .foregroundStyle(p.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 6) {
                    Text(event.verdict.uppercased())
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(color)
                    if let risk = event.riskScore {
                        Text(String(format: "risk %.2f", risk))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(p.ghost)
                    }
                }
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 6)
        .help(event.reason)
    }
}

private struct FeedRow: View {
    let event: DecisionEvent
    let p: Palette

    private var color: Color {
        event.verdict == "deny" ? GateState.red.color : GateState.yellow.color
    }

    var body: some View {
        HStack(spacing: 12) {
            Text(event.timestamp, format: .dateTime.hour().minute())
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(p.ghost)
            Rectangle().fill(color).frame(width: 2, height: 22).cornerRadius(1)
            VStack(alignment: .leading, spacing: 1) {
                Text(event.toolName)
                    .font(.system(size: 12))
                    .foregroundStyle(p.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 6) {
                    Text(event.verdict.uppercased())
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(color)
                    if let risk = event.riskScore {
                        Text(String(format: "risk %.2f", risk))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(p.ghost)
                    }
                }
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 7)
        .help(event.reason)
    }
}
