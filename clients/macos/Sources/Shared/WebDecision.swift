import Foundation

/// What the filter does with what `vaara check` printed, or failed to print.
///
/// `vaara check` prints one JSON object: `decision`, `allowed`, `reason`,
/// `action_id`. The filter used to allow when that object did not arrive: a
/// crash, a missing binary, a non-JSON answer and an object without `allowed`
/// all replied allow, so a broken install looked like a permissive policy.
/// Every other Vaara surface refuses what it cannot decide; now so does this
/// one, and the reply says the engine was unavailable rather than that policy
/// refused. An `escalate` decision is carried as such, so the extension
/// pauses the flow for a person instead of dropping it.
public struct WebDecision: Equatable {
    public let allowed: Bool
    /// nil for allow, "escalate" for an escalation, else why it was refused.
    public let reason: String?
    public let actionId: String?

    public init(allowed: Bool, reason: String?, actionId: String?) {
        self.allowed = allowed
        self.reason = reason
        self.actionId = actionId
    }

    /// A refusal because the engine did not answer carries this prefix on
    /// the XPC reply, so the extension logs an unavailable engine and not a
    /// policy decision.
    public static let unavailablePrefix = "unavailable: "

    public static func unavailable(_ why: String) -> WebDecision {
        WebDecision(allowed: false, reason: unavailablePrefix + why, actionId: nil)
    }

    public var engineUnavailable: Bool {
        reason?.hasPrefix(Self.unavailablePrefix) ?? false
    }

    /// `output` is what `vaara check` wrote to stdout, `terminationStatus`
    /// how it exited (nil when it never ran), `launchError` why it never ran.
    public static func parse(output: Data?, terminationStatus: Int32?,
                             launchError: String? = nil) -> WebDecision {
        if let launchError {
            return unavailable("vaara check could not run: \(launchError)")
        }
        guard let status = terminationStatus else {
            return unavailable("vaara check never ran")
        }
        guard let data = output, !data.isEmpty,
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return unavailable("vaara check printed no decision (exit \(status))")
        }
        guard let allowed = object["allowed"] as? Bool else {
            return unavailable("vaara check answered without `allowed` (exit \(status))")
        }
        let actionId = object["action_id"] as? String
        if allowed {
            return WebDecision(allowed: true, reason: nil, actionId: actionId)
        }
        if (object["decision"] as? String) == "escalate" {
            return WebDecision(allowed: false, reason: "escalate", actionId: actionId)
        }
        let reason = (object["reason"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        return WebDecision(allowed: false, reason: reason ?? "blocked by policy", actionId: actionId)
    }
}
