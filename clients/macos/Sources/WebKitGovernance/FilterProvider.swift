import NetworkExtension
import OSLog
// Shared is a separate module only under SwiftPM. The XcodeGen and
// Homebrew builds compile Sources/Shared into this target directly.
#if canImport(Shared)
import Shared
#endif

let log = OSLog(subsystem: "io.vaara.menubar", category: "NetworkFilter")

/// System-wide network content filter for AI traffic.
///
/// Runs as a `NEFilterDataProvider` system extension, so it sees flows from
/// every application on the machine, including Safari, Mail, Orion and any
/// other WKWebView host, with no browser extension installed. That reach is
/// the point: Apple does not expose the hooks a Safari extension would need.
///
/// Scope, stated plainly: this governs flows by hostname. It is not a WebKit
/// integration and has no knowledge of page content or the web view itself.
final class FilterProvider: NEFilterDataProvider {

    // ── Lifecycle ─────────────────────────────────────────────────────────

    override func startFilter(completionHandler: @escaping (Error?) -> Void) {
        // Only ask the kernel for flows to hosts we actually govern. Every
        // other flow is never handed to us, which keeps the common path free.
        let rules: [NEFilterRule] = GovernedHosts.all().compactMap { host in
            let net = NWHostEndpoint(hostname: host, port: "443")
            guard let network = NENetworkRule(remoteNetwork: net,
                                              remotePrefix: 0,
                                              localNetwork: nil,
                                              localPrefix: 0,
                                              protocol: .TCP,
                                              direction: .outbound) as NENetworkRule?
            else { return nil }
            return NEFilterRule(networkRule: network, action: .filterData)
        }

        let settings = NEFilterSettings(rules: rules, defaultAction: .allow)
        apply(settings) { error in
            if let error {
                os_log(.error, log: log, "failed to apply filter settings: %{public}s",
                       error.localizedDescription)
            } else {
                os_log(.info, log: log, "filter started, %{public}d governed hosts",
                       rules.count)
            }
            completionHandler(error)
        }
    }

    override func stopFilter(with reason: NEProviderStopReason,
                             completionHandler: @escaping () -> Void) {
        os_log(.info, log: log, "filter stopped, reason %{public}d", reason.rawValue)
        completionHandler()
    }

    // ── Flow-level filtering ──────────────────────────────────────────────

    override func handleNewFlow(_ flow: NEFilterFlow) -> NEFilterNewFlowVerdict {
        guard let host = hostname(for: flow) else {
            // No hostname available. Allowed, and said out loud rather than
            // dropped silently, because an unattributable flow is exactly the
            // case an operator needs to know about.
            os_log(.info, log: log, "allow: flow with no resolvable hostname")
            return .allow()
        }

        guard GovernedHosts.isGoverned(host) else {
            return .allow()
        }

        let url = flow.url?.absoluteString ?? host
        os_log(.info, log: log, "governed flow: %{public}s", url)

        switch VaaraPolicyClient.shared.decide(host: host, url: url) {
        case .allow:
            os_log(.info, log: log, "allow: %{public}s", url)
            return .allow()

        case .deny(let reason):
            os_log(.info, log: log, "deny: %{public}s (%{public}s)", url, reason)
            return .drop()

        case .escalate:
            // Pause hands the flow to the app for a human decision. The app
            // resumes it with `resumeFlow(_:with:)` once the person answers.
            os_log(.info, log: log, "escalate: %{public}s", url)
            return .pause()

        case .unavailable(let reason):
            // The policy engine is unreachable or slow. This is a governed
            // host, so the decision matters: fail closed rather than quietly
            // approving traffic nobody evaluated.
            os_log(.error, log: log,
                   "deny (engine unavailable): %{public}s (%{public}s)", url, reason)
            return .drop()
        }
    }

    /// Best available hostname for a flow.
    ///
    /// `NEFilterFlow.url` is populated for flows the system could attribute to
    /// a URL. Socket flows carry only an endpoint, so fall back to that rather
    /// than treating the flow as unattributable.
    private func hostname(for flow: NEFilterFlow) -> String? {
        if let host = flow.url?.host, !host.isEmpty {
            return host
        }
        if let socketFlow = flow as? NEFilterSocketFlow,
           let endpoint = socketFlow.remoteEndpoint as? NWHostEndpoint,
           !endpoint.hostname.isEmpty {
            return endpoint.hostname
        }
        return nil
    }

    // ── Data-level filtering ──────────────────────────────────────────────
    //
    // Not implemented. The verdict is taken at flow level, on hostname, before
    // any payload moves. These overrides exist because the superclass requires
    // them once `.filterData` is requested; they inspect nothing and say so.
    // Payload inspection would mean terminating TLS, which this extension
    // deliberately does not do.

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
