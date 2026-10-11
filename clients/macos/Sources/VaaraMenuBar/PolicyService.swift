import Foundation
// Shared is a separate module only under SwiftPM. The XcodeGen and
// Homebrew builds compile Sources/Shared into this target directly.
#if canImport(Shared)
import Shared
#endif

/// XPC service listener — runs in the main Vaara app process.
/// The Network Extension connects here to request policy decisions.
/// Uses the `vaara` CLI (installed via Homebrew) for policy evaluation.
final class PolicyServiceDelegate: NSObject, NSXPCListenerDelegate, VaaraPolicyService {

    private let listener: NSXPCListener
    private let trailDB: String

    override init() {
        // App-group prefixed. A sandboxed system extension cannot reach a bare
        // mach service name, so the extension's connection never resolved.
        self.listener = NSXPCListener(machServiceName: vaaraPolicyMachService)
        self.trailDB = NSString(string: "~/.vaara/trail/audit.db").expandingTildeInPath
        super.init()
        listener.delegate = self
        listener.resume()
    }

    // ── NSXPCListenerDelegate ─────────────────────────────────────────

    func listener(_ listener: NSXPCListener,
                  shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: VaaraPolicyService.self)
        newConnection.exportedObject = self
        newConnection.resume()
        return true
    }

    // ── VaaraPolicyService ─────────────────────────────────────────────

    func evaluate(host: String, url: String, reply: @escaping (Bool, String?) -> Void) {
        // Use the `vaara check` CLI to evaluate this action.
        // This runs the same Vaara pipeline as everywhere else. What it
        // printed, or failed to print, is read by WebDecision, which refuses
        // anything that is not a decision: a crash, a missing binary and a
        // non-JSON answer are all replied as the engine being unavailable,
        // never as allow.
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = [
            "vaara", "check",
            "--tool", "web.navigate",
            "--param", "host=\(host)",
            "--param", "url=\(url)",
            "--agent", "webkit",
            "--db", trailDB,
        ]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()

        let decision: WebDecision
        do {
            try task.run()
            // Read before waiting: a child that fills the pipe would otherwise
            // block on write while this side blocks on exit.
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            task.waitUntilExit()
            decision = WebDecision.parse(output: data, terminationStatus: task.terminationStatus)
        } catch {
            decision = WebDecision.parse(output: nil, terminationStatus: nil,
                                         launchError: error.localizedDescription)
        }
        reply(decision.allowed, decision.reason)
    }

    func recordOutcome(actionId: String, severity: Double, description: String) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = [
            "vaara", "outcome",
            "--action-id", actionId,
            "--severity", String(severity),
            "--db", trailDB,
        ]
        try? task.run()
    }

    func notifyInteraction(host: String, url: String, title: String, action: String) {
        // Log the interaction to the audit trail as a check call.
        // Future: this can feed into a risk score that modulates the filter.
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = [
            "vaara", "check",
            "--tool", "web.ui_interaction",
            "--agent", "webkit",
            "--param", "host=\(host)",
            "--param", "url=\(url)",
            "--param", "action=\(action)",
            "--param", "title=\(title)",
            "--db", trailDB,
        ]
        try? task.run()
    }
}
