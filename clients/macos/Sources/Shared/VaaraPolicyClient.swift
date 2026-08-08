import Foundation
import OSLog

private let clientLog = OSLog(subsystem: "io.vaara.menubar", category: "PolicyClient")

public enum PolicyDecision {
    case allow
    case deny(reason: String)
    case escalate
    /// The policy engine could not be reached or did not answer in time.
    ///
    /// Kept distinct from `.allow` on purpose. The old code returned `.allow`
    /// for both, so a filter that had lost its policy engine looked exactly
    /// like one that had approved the traffic. The caller decides what to do
    /// with this; it must not be silently indistinguishable from approval.
    case unavailable(reason: String)
}

/// Bridge from the network filter extension to the Vaara pipeline in the app.
///
/// Shared source: compiled into both targets. It previously lived only in the
/// app target while the extension referenced it, which is why the extension
/// could not compile.
public final class VaaraPolicyClient {

    public static let shared = VaaraPolicyClient()

    private var connection: NSXPCConnection?
    private let lock = NSLock()
    private let timeout: TimeInterval = 2.0

    private init() {}

    // ── Connection ────────────────────────────────────────────────────────

    /// Lazily connect, and reconnect if a previous connection was invalidated.
    private func service() -> VaaraPolicyService? {
        lock.lock()
        defer { lock.unlock() }

        if connection == nil {
            let conn = NSXPCConnection(machServiceName: vaaraPolicyMachService)
            conn.remoteObjectInterface = NSXPCInterface(with: VaaraPolicyService.self)
            conn.invalidationHandler = { [weak self] in
                self?.clearConnection()
            }
            conn.interruptionHandler = { [weak self] in
                self?.clearConnection()
            }
            conn.resume()
            connection = conn
        }

        return connection?.remoteObjectProxyWithErrorHandler { error in
            os_log(.error, log: clientLog, "xpc error: %{public}s",
                   error.localizedDescription)
        } as? VaaraPolicyService
    }

    private func clearConnection() {
        lock.lock()
        connection = nil
        lock.unlock()
    }

    // ── Decisions ─────────────────────────────────────────────────────────

    /// Ask the app whether this flow should be allowed.
    ///
    /// Returns `.unavailable` rather than `.allow` when the engine cannot be
    /// reached or does not answer within `timeout`.
    public func decide(host: String, url: String) -> PolicyDecision {
        guard let service = service() else {
            return .unavailable(reason: "policy engine not reachable")
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: PolicyDecision = .unavailable(reason: "no reply")

        service.evaluate(host: host, url: url) { allowed, reason in
            if allowed {
                result = .allow
            } else if reason == "escalate" {
                result = .escalate
            } else {
                result = .deny(reason: reason ?? "blocked by policy")
            }
            semaphore.signal()
        }

        if semaphore.wait(timeout: .now() + timeout) == .timedOut {
            return .unavailable(reason: "policy engine timed out after \(timeout)s")
        }
        return result
    }

    /// Record that a user interacted with a governed site. Context only.
    public func notifyInteraction(host: String, url: String, title: String, action: String) {
        service()?.notifyInteraction(host: host, url: url, title: title, action: action)
    }
}
