import Foundation

/// Hosts whose outbound traffic the filter governs.
///
/// Shared source, compiled into both targets, so the app's settings UI and the
/// extension's filter read exactly the same list.
///
/// Two families, and the distinction is the one the old hardcoded list got
/// wrong. Governing only the API endpoints meant a browser talking to a chat
/// UI was never seen, so a claim to cover Safari or Mail could not hold.
public enum GovernedHosts {

    /// Programmatic model endpoints. Traffic here is an SDK, a CLI, or an agent.
    public static let apiHosts: Set<String> = [
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.fireworks.ai",
        "api.melious.ai",
        "api.together.xyz",
        "api.deepseek.com",
        "api.mistral.ai",
        "router.huggingface.co",
        "api.x.ai",
        "api.cohere.ai",
        "api.perplexity.ai",
        "openrouter.ai",
    ]

    /// Chat front ends. Traffic here is a person in Safari, Mail, Orion, or any
    /// other WKWebView host application.
    public static let webUIHosts: Set<String> = [
        "chatgpt.com",
        "chat.openai.com",
        "claude.ai",
        "gemini.google.com",
        "chat.deepseek.com",
        "chat.mistral.ai",
        "perplexity.ai",
        "copilot.microsoft.com",
        "grok.com",
        "poe.com",
    ]

    /// Key under which operator additions are stored in the shared app group.
    public static let userAdditionsKey = "vaara.governedHosts.userAdditions"

    /// Hosts the operator added from the Vaara app's settings.
    ///
    /// The previous code carried a comment promising this and then hardcoded a
    /// `private let`, so the promise could not be kept. Reading it from the
    /// shared app-group defaults makes app and extension agree.
    public static func userAdditions() -> Set<String> {
        guard let defaults = UserDefaults(suiteName: vaaraAppGroup),
              let stored = defaults.array(forKey: userAdditionsKey) as? [String]
        else { return [] }
        return Set(stored.map { $0.lowercased() })
    }

    /// Persist the operator's additions. Called by the app's settings UI.
    public static func setUserAdditions(_ hosts: Set<String>) {
        guard let defaults = UserDefaults(suiteName: vaaraAppGroup) else { return }
        defaults.set(Array(hosts).sorted(), forKey: userAdditionsKey)
    }

    /// Every governed host: built-ins plus operator additions.
    public static func all() -> Set<String> {
        apiHosts.union(webUIHosts).union(userAdditions())
    }

    /// True when this hostname is governed.
    ///
    /// Matches the host itself and any subdomain of it, because exact string
    /// equality missed regional and sharded endpoints entirely.
    public static func isGoverned(_ hostname: String) -> Bool {
        let host = hostname.lowercased()
        for governed in all() {
            if host == governed || host.hasSuffix("." + governed) {
                return true
            }
        }
        return false
    }
}
