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
    func refresh() {
        NEFilterManager.shared().loadFromPreferences { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                if let error {
                    self.state = .failed(error.localizedDescription)
                    return
                }
                if NEFilterManager.shared().providerConfiguration == nil {
                    self.state = .notInstalled
                } else {
                    self.state = NEFilterManager.shared().isEnabled ? .active : .installedDisabled
                }
            }
        }
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
