import Foundation

/// App group shared by the menu-bar app and the network filter extension.
///
/// A sandboxed system extension may only reach a mach service whose name is
/// prefixed with an app group both sides hold an entitlement for. The bare
/// name this used to carry (`io.vaara.policyengine`) is unreachable from the
/// sandbox, so the extension's connection would never have resolved.
public let vaaraAppGroup = "group.io.vaara"

/// Mach service the app vends and the extension connects to.
public let vaaraPolicyMachService = "\(vaaraAppGroup).policyengine"

/// XPC protocol between the network filter extension and the Vaara app.
///
/// Shared source: this file is compiled into BOTH targets. It previously lived
/// only in the app target while the extension referenced it, so the extension
/// could not compile.
@objc public protocol VaaraPolicyService {
    /// Evaluate whether a flow to `host` with `url` should be allowed.
    /// - allowed: true if the flow passes policy
    /// - reason: nil for allow, string for deny, "escalate" for escalation
    func evaluate(host: String, url: String, reply: @escaping (_ allowed: Bool, _ reason: String?) -> Void)

    /// Record the outcome of a previously evaluated flow.
    func recordOutcome(actionId: String, severity: Double, description: String)

    /// Called by AccessibilityObserver when a user interacts with an AI site.
    func notifyInteraction(host: String, url: String, title: String, action: String)
}
