import Foundation

/// Bridge between the Network Extension and the Vaara pipeline running in the main app.
/// Uses XPC to request policy decisions and record outcomes.
enum PolicyDecision {
    case allow
    case deny(reason: String)
    case escalate
}

final class VaaraPolicyClient {

    static let shared = VaaraPolicyClient()

    private var connection: NSXPCConnection?

    private init() {
        setupXPC()
    }

    private func setupXPC() {
        let conn = NSXPCConnection(machServiceName: "io.vaara.policyengine")
        conn.remoteObjectInterface = NSXPCInterface(with: VaaraPolicyService.self)
        conn.resume()
        self.connection = conn
    }

    /// Ask the main Vaara app whether this flow should be allowed.
    func decide(host: String, url: String) -> PolicyDecision {
        guard let service = connection?.remoteObjectProxy as? VaaraPolicyService else {
            return .allow
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: PolicyDecision = .allow

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

        _ = semaphore.wait(timeout: .now() + 2.0)
        return result
    }

    /// Called by AccessibilityObserver when a user interaction is detected
    /// on an AI site. Records context to the audit trail.
    func notifyInteraction(host: String, url: String, title: String, action: String) {
        guard let service = connection?.remoteObjectProxy as? VaaraPolicyService else { return }
        service.notifyInteraction(host: host, url: url, title: title, action: action)
    }
}
