import Cocoa
import OSLog

private let log = OSLog(subsystem: "io.vaara.menubar", category: "Accessibility")

/// Known AI web interfaces we can detect and observe.
private let aiSites: Set<String> = [
    "chatgpt.com",
    "claude.ai",
    "gemini.google.com",
    "chat.deepseek.com",
    "chat.mistral.ai",
    "perplexity.ai",
]

/// Detects AI interactions in WebKit-based apps using the Accessibility API.
/// Runs as an observer in the main app process.
final class AccessibilityObserver {

    private var runLoopSource: CFRunLoopSource?
    private var isRunning = false

    /// Start observing UI interactions. Requires Accessibility permission.
    func start() {
        guard checkPermission() else {
            os_log(.info, log: log, "accessibility permission not granted")
            return
        }

        let options: NSDictionary = [
            kAXTrustedCheckOptionPrompt.takeRetainedValue(): true,
        ]
        guard AXIsProcessTrustedWithOptions(options) else {
            os_log(.info, log: log, "not trusted by accessibility API")
            return
        }

        // Observe global app focus changes.
        let pid = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        var obs: AXObserver?
        let createErr = AXObserverCreate(pid, { observer, element, notification, refcon in
            let selfPtr = Unmanaged<AccessibilityObserver>.fromOpaque(refcon!).takeUnretainedValue()
            selfPtr.handleNotification(element: element, notification: notification as String)
        }, &obs)

        guard createErr == .success, let observer = obs else {
            os_log(.error, log: log, "failed to create observer: %d", createErr.rawValue)
            return
        }

        // Watch for focused UI element changes in the frontmost app.
        let appElement = AXUIElementCreateApplication(
            NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        )
        AXObserverAddNotification(observer, appElement,
                                  kAXFocusedUIElementChangedNotification as CFString,
                                  self)

        runLoopSource = AXObserverGetRunLoopSource(observer)
        if let source = runLoopSource {
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        }

        isRunning = true
        os_log(.info, log: log, "accessibility observer started")
    }

    func stop() {
        guard isRunning, let source = runLoopSource else { return }
        CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        isRunning = false
    }

    // ── Permission ────────────────────────────────────────────────────

    private func checkPermission() -> Bool {
        AXIsProcessTrusted()
    }

    // ── Notification handler ──────────────────────────────────────────

    private func handleNotification(element: AXUIElement, notification: String) {
        // Get the URL of the frontmost browser tab.
        guard let url = currentBrowserURL() else { return }
        guard let host = URL(string: url)?.host else { return }

        // Check if this is an AI site.
        guard aiSites.contains(where: { host.contains($0) }) else { return }

        // Get the page title for context.
        let title = currentBrowserTitle() ?? ""

        // Detect if the user is interacting with the chat input.
        let isComposing = isComposingMessage(element: element)

        os_log(.info, log: log,
               "AI interaction detected: %{public}s — composing: %{public}s",
               host, String(isComposing))

        // Notify the Vaara pipeline when a user is about to send a message.
        if isComposing {
            VaaraPolicyClient.shared.notifyInteraction(
                host: host,
                url: url,
                title: title,
                action: "compose"
            )
        }
    }

    // ── Safari / browser introspection ────────────────────────────────

    private func currentBrowserURL() -> String? {
        let app = NSWorkspace.shared.frontmostApplication
        guard let pid = app?.processIdentifier else { return nil }

        let appElement = AXUIElementCreateApplication(pid)

        // Try to get the URL via the browser's accessibility hierarchy.
        // Safari exposes the current URL in the "value" attribute of the URL field.
        var value: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appElement,
                                                    "AXURL" as CFString,
                                                    &value)
        if result == .success, let url = value as? String {
            return url
        }

        // Fallback: check window title for known patterns.
        var window: CFTypeRef?
        AXUIElementCopyAttributeValue(appElement,
                                       kAXMainWindowAttribute as CFString,
                                       &window)
        if let windowElement = window as! AXUIElement? {
            var title: CFTypeRef?
            AXUIElementCopyAttributeValue(windowElement,
                                           kAXTitleAttribute as CFString,
                                           &title)
            return title as? String
        }

        return nil
    }

    private func currentBrowserTitle() -> String? {
        let app = NSWorkspace.shared.frontmostApplication
        guard let pid = app?.processIdentifier else { return nil }

        var window: CFTypeRef?
        AXUIElementCopyAttributeValue(
            AXUIElementCreateApplication(pid),
            kAXMainWindowAttribute as CFString,
            &window
        )

        guard let windowElement = window as! AXUIElement? else { return nil }
        var title: CFTypeRef?
        AXUIElementCopyAttributeValue(windowElement,
                                       kAXTitleAttribute as CFString,
                                       &title)
        return title as? String
    }

    /// Heuristic: detect if the currently focused element looks like
    /// a chat message input field (multiline text area on an AI site).
    private func isComposingMessage(element: AXUIElement) -> Bool {
        var role: CFTypeRef?
        AXUIElementCopyAttributeValue(element,
                                       kAXRoleAttribute as CFString,
                                       &role)
        guard let roleStr = role as? String else { return false }

        // Chat inputs are typically text areas or rich text fields.
        return roleStr == "AXTextArea" || roleStr == "AXComboBox"
    }
}
