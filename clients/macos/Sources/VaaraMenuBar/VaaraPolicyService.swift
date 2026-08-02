import Foundation

/// XPC protocol between the Network Extension and the Vaara main app.
@objc protocol VaaraPolicyService {
    /// Evaluate whether a flow to `host` with `url` should be allowed.
    /// - allowed: true if the flow passes policy
    /// - reason: nil for allow, string for deny, "escalate" for escalation
    func evaluate(host: String, url: String, reply: @escaping (_ allowed: Bool, _ reason: String?) -> Void)

    /// Record the outcome of a previously evaluated flow.
    func recordOutcome(actionId: String, severity: Double, description: String)

    /// Called by AccessibilityObserver when user interacts with an AI site.
    func notifyInteraction(host: String, url: String, title: String, action: String)
}
