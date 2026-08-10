import Foundation
import NetworkExtension
import OSLog
import SystemExtensions

private let extLog = OSLog(subsystem: "io.vaara.menubar", category: "SystemExtension")

/// Installs and enables the network filter system extension.
///
/// Nothing previously did this. The extension target existed, but no code ever
/// submitted an `OSSystemExtensionRequest` or configured `NEFilterManager`, so
/// even a correctly built extension would never have run. macOS requires both
/// steps: activation installs the bundle, and the filter configuration enables
/// it. The user is prompted for each.
@MainActor
final class SystemExtensionManager: NSObject, ObservableObject {

    enum State: Equatable {
        case unknown
        case notInstalled
        case awaitingUserApproval
        case installedDisabled
        case active
        case failed(String)
    }

    static let shared = SystemExtensionManager()

    @Published private(set) var state: State = .unknown

    private let extensionIdentifier = "io.vaara.webkit-governance"

    // ── Public API ────────────────────────────────────────────────────────

    /// Reflect the current filter configuration into `state`.
    ///
    /// The saved `NEFilterManager` preference is not evidence that anything is
    /// filtering. It survives the extension being removed, by an OS update, by
    /// `systemextensionsctl reset`, or by the user deleting the app, and it is
    /// written before macOS ever agrees to load the provider. Reporting
    /// "filtering AI-bound traffic" from that alone is a status display that
    /// fails open, which is the one direction this product must never fail.
    ///
    /// So the preference is only allowed to downgrade the verdict. The claim
    /// that the filter is live has to come from the system's own list of
    /// installed extensions.
    func refresh() {
        NEFilterManager.shared().loadFromPreferences { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                if let error {
                    self.state = .failed(error.localizedDescription)
                    return
                }
                let configured = NEFilterManager.shared().providerConfiguration != nil
                let enabled = NEFilterManager.shared().isEnabled
                guard configured else {
                    self.state = .notInstalled
                    return
                }
                self.confirmInstalled { installed in
                    if !installed {
                        // Configured but absent: say so rather than claiming
                        // a filter that cannot possibly be running.
                        self.state = .notInstalled
                    } else {
                        self.state = enabled ? .active : .installedDisabled
                    }
                }
            }
        }
    }

    /// Ask the system whether our extension is actually installed.
    ///
    /// A properties request also completes through
    /// `OSSystemExtensionRequestDelegate`, and this class's delegate reacts to
    /// completion by enabling the filter, so a refresh sharing that delegate
    /// would turn the filter on as a side effect of looking at it. The query
    /// gets its own delegate object for that reason.
    private func confirmInstalled(_ done: @escaping @MainActor (Bool) -> Void) {
        let probe = InstalledProbe(done: done)
        let request = OSSystemExtensionRequest.propertiesRequest(
            forExtensionWithIdentifier: extensionIdentifier,
            queue: .main
        )
        request.delegate = probe
        probe.retain = probe          // live until the delegate answers
        OSSystemExtensionManager.shared.submitRequest(request)
    }

    /// Install the extension, then enable the content filter.
    func activate() {
        let request = OSSystemExtensionRequest.activationRequest(
            forExtensionWithIdentifier: extensionIdentifier,
            queue: .main
        )
        request.delegate = self
        state = .awaitingUserApproval
        OSSystemExtensionManager.shared.submitRequest(request)
    }

    /// Disable the filter and remove the extension.
    func deactivate() {
        NEFilterManager.shared().isEnabled = false
        NEFilterManager.shared().saveToPreferences { _ in
            let request = OSSystemExtensionRequest.deactivationRequest(
                forExtensionWithIdentifier: self.extensionIdentifier,
                queue: .main
            )
            request.delegate = self
            OSSystemExtensionManager.shared.submitRequest(request)
        }
    }

    // ── Filter configuration ──────────────────────────────────────────────

    /// Enable the content filter once the extension is installed. Activation
    /// alone does not start filtering.
    private func enableFilter() {
        NEFilterManager.shared().loadFromPreferences { [weak self] error in
            guard let self else { return }
            if let error {
                Task { @MainActor in self.state = .failed(error.localizedDescription) }
                return
            }

            if NEFilterManager.shared().providerConfiguration == nil {
                let config = NEFilterProviderConfiguration()
                config.filterSockets = true
                config.filterPackets = false
                NEFilterManager.shared().providerConfiguration = config
                NEFilterManager.shared().localizedDescription = "Vaara"
            }
            NEFilterManager.shared().isEnabled = true

            NEFilterManager.shared().saveToPreferences { error in
                Task { @MainActor in
                    if let error {
                        self.state = .failed(error.localizedDescription)
                    } else {
                        self.state = .active
                        os_log(.info, log: extLog, "content filter enabled")
                    }
                }
            }
        }
    }
}

/// Answers one question: does the system have this extension installed?
///
/// Kept apart from `SystemExtensionManager`'s own delegate so a read cannot
/// trip the write path. `foundProperties` arrives before completion; an empty
/// list, or a failure, both mean "not installed" and both must resolve, or a
/// refresh would hang the status line on its previous value.
private final class InstalledProbe: NSObject, OSSystemExtensionRequestDelegate {
    private let done: @MainActor (Bool) -> Void
    private var answered = false
    var retain: InstalledProbe?

    init(done: @escaping @MainActor (Bool) -> Void) {
        self.done = done
    }

    private func answer(_ installed: Bool) {
        guard !answered else { return }
        answered = true
        let done = self.done
        Task { @MainActor in
            done(installed)
            self.retain = nil
        }
    }

    func request(_ request: OSSystemExtensionRequest,
                 foundProperties properties: [OSSystemExtensionProperties]) {
        answer(properties.contains { $0.isEnabled || $0.isAwaitingUserApproval } )
    }

    func request(_ request: OSSystemExtensionRequest,
                 didFinishWithResult result: OSSystemExtensionRequest.Result) {
        answer(false)   // no properties arrived, so nothing is installed
    }

    func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        answer(false)
    }

    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {}

    func request(_ request: OSSystemExtensionRequest,
                 actionForReplacingExtension existing: OSSystemExtensionProperties,
                 withExtension new: OSSystemExtensionProperties)
    -> OSSystemExtensionRequest.ReplacementAction {
        return .cancel
    }
}

// ── OSSystemExtensionRequestDelegate ──────────────────────────────────────

extension SystemExtensionManager: OSSystemExtensionRequestDelegate {

    nonisolated func request(_ request: OSSystemExtensionRequest,
                             didFinishWithResult result: OSSystemExtensionRequest.Result) {
        Task { @MainActor in
            switch result {
            case .completed:
                os_log(.info, log: extLog, "extension activated")
                self.enableFilter()
            case .willCompleteAfterReboot:
                self.state = .installedDisabled
                os_log(.info, log: extLog, "extension activates after reboot")
            @unknown default:
                self.state = .failed("unknown activation result")
            }
        }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest,
                             didFailWithError error: Error) {
        Task { @MainActor in
            os_log(.error, log: extLog, "activation failed: %{public}s",
                   error.localizedDescription)
            self.state = .failed(error.localizedDescription)
        }
    }

    nonisolated func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        Task { @MainActor in
            os_log(.info, log: extLog, "awaiting approval in System Settings")
            self.state = .awaitingUserApproval
        }
    }

    /// Called when a different build of the extension is already installed.
    /// Always take the one shipped inside this app bundle.
    nonisolated func request(_ request: OSSystemExtensionRequest,
                             actionForReplacingExtension existing: OSSystemExtensionProperties,
                             withExtension new: OSSystemExtensionProperties)
    -> OSSystemExtensionRequest.ReplacementAction {
        return .replace
    }
}
