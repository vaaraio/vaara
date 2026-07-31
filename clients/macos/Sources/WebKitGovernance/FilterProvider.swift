import NetworkExtension
import OSLog

let log = OSLog(subsystem: "io.vaara.menubar", category: "WebKitFilter")

/// Hosts whose outbound traffic we intercept and govern.
/// Users can extend this via the Vaara app settings.
private let governedHosts: Set<String> = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.fireworks.ai",
    "api.melious.ai",
    "api.together.xyz",
    "api.deepseek.com",
    "api.mistral.ai",
    "router.huggingface.co",
]

class FilterProvider: NEFilterDataProvider {

    // ── Flow-level filtering ──────────────────────────────────────────────

    override func handleNewFlow(_ flow: NEFilterFlow) -> NEFilterNewFlowVerdict {
        guard let host = flow.hostname else {
            return .allow()
        }

        guard governedHosts.contains(host) else {
            return .allow()
        }

        let url = flow.url?.absoluteString ?? host
        os_log(.info, log: log, "flow to governed host: %{public}s", url)

        // Communicate with the Vaara main app for a policy decision.
        // The app runs the InterceptionPipeline and returns allow/deny.
        let decision = VaaraPolicyClient.shared.decide(host: host, url: url)

        switch decision {
        case .allow:
            os_log(.info, log: log, "allow: %{public}s", url)
            return .allow()
        case .deny(let reason):
            os_log(.info, log: log, "deny: %{public}s — %{public}s", url, reason)
            return .drop()
        case .escalate:
            os_log(.info, log: log, "escalate: %{public}s", url)
            return .pause()
        }
    }

    // ── Data-level filtering (inspects payload) ───────────────────────────

    override func handleInboundData(from flow: NEFilterFlow,
                                     readBytesStartOffset offset: Int,
                                     readBytes: Data) -> NEFilterDataVerdict {
        return .allow()
    }

    override func handleOutboundData(from flow: NEFilterFlow,
                                      readBytesStartOffset offset: Int,
                                      readBytes: Data) -> NEFilterDataVerdict {
        return .allow()
    }
}
