import Cocoa
import OSLog
// Shared is a separate module only under SwiftPM. The XcodeGen and
// Homebrew builds compile Sources/Shared into this target directly.
#if canImport(Shared)
import Shared
#endif

private let log = OSLog(subsystem: "io.vaara.menubar", category: "Accessibility")

private let aiSiteHosts: [String] = [
    "chatgpt.com",
    "claude.ai",
    "gemini.google.com",
    "chat.deepseek.com",
    "chat.mistral.ai",
    "perplexity.ai",
    "copilot.microsoft.com",
    "chat.openai.com",
]

final class AccessibilityObserver {

    static let shared = AccessibilityObserver()

    private var observers: [pid_t: AXObserver] = [:]
    private var isRunning = false
    private var focusObserver: NSObjectProtocol?

    func start() {
        guard checkPermission() else {
            os_log(.info, log: log, "accessibility permission not granted")
            return
        }

        let opts: NSDictionary = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): true]
        guard AXIsProcessTrustedWithOptions(opts) else {
            os_log(.info, log: log, "not trusted by accessibility API")
            return
        }

        if let app = NSWorkspace.shared.frontmostApplication {
            attachTo(pid: app.processIdentifier)
        }

        focusObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil, queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
            self?.handleAppSwitch(pid: app.processIdentifier)
        }

        isRunning = true
        os_log(.info, log: log, "accessibility observer started")
    }

    func stop() {
        for (pid, obs) in observers {
            detachObserver(obs, pid: pid)
        }
        observers.removeAll()

        if let token = focusObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(token)
            focusObserver = nil
        }
        isRunning = false
    }

    // ── App switching ──────────────────────────────────────────────

    private func handleAppSwitch(pid: pid_t) {
        guard isBrowser(pid: pid) else { return }
        if observers[pid] == nil {
            attachTo(pid: pid)
        }
    }

    private func isBrowser(pid: pid_t) -> Bool {
        guard let app = NSRunningApplication(processIdentifier: pid) else { return false }
        let bundle = app.bundleIdentifier ?? ""
        let browserBundles = [
            "com.apple.Safari",
            "com.google.Chrome",
            "org.mozilla.firefox",
            "com.microsoft.edgemac",
            "com.brave.Browser",
            "company.thebrowser.Arc",
        ]
        return browserBundles.contains(bundle)
    }

    // ── AXObserver lifecycle ───────────────────────────────────────

    private func attachTo(pid: pid_t) {
        var obs: AXObserver?
        let err = AXObserverCreate(pid, { observer, element, notification, refcon in
            guard let refcon = refcon else { return }
            let ptr = Unmanaged<AccessibilityObserver>.fromOpaque(refcon).takeUnretainedValue()
            ptr.onFocusChange(element: element, notification: notification as String)
        }, &obs)

        guard err == .success, let observer = obs else {
            os_log(.error, log: log, "AXObserverCreate failed pid=%d err=%d", pid, err.rawValue)
            return
        }

        let appEl = AXUIElementCreateApplication(pid)
        AXObserverAddNotification(observer, appEl,
            kAXFocusedUIElementChangedNotification as CFString,
            Unmanaged.passUnretained(self).toOpaque())
        AXObserverAddNotification(observer, appEl,
            kAXFocusedWindowChangedNotification as CFString,
            Unmanaged.passUnretained(self).toOpaque())

        let source = AXObserverGetRunLoopSource(observer)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)

        observers[pid] = observer
        os_log(.info, log: log, "attached AXObserver pid=%d", pid)
    }

    private func detachObserver(_ observer: AXObserver, pid: pid_t) {
        let appEl = AXUIElementCreateApplication(pid)
        AXObserverRemoveNotification(observer, appEl,
            kAXFocusedUIElementChangedNotification as CFString)
        AXObserverRemoveNotification(observer, appEl,
            kAXFocusedWindowChangedNotification as CFString)
        let source = AXObserverGetRunLoopSource(observer)
        CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
    }

    // ── Permission ─────────────────────────────────────────────────

    private func checkPermission() -> Bool {
        AXIsProcessTrusted()
    }

    // ── Notification handler ───────────────────────────────────────

    private func onFocusChange(element: AXUIElement, notification: String) {
        guard let url = currentBrowserURL() else { return }
        guard let host = URL(string: url)?.host else { return }
        guard isAISite(host: host) else { return }

        let title = currentBrowserTitle() ?? ""
        let composing = isComposingMessage(element: element)

        os_log(.info, log: log,
               "AI interaction: %{public}s composing=%{public}s",
               host, String(composing))

        if composing {
            VaaraPolicyClient.shared.notifyInteraction(
                host: host, url: url, title: title, action: "compose"
            )
        }
    }

    private func isAISite(host: String) -> Bool {
        aiSiteHosts.contains { site in
            host == site || host.hasSuffix("." + site)
        }
    }

    // ── Browser URL extraction ─────────────────────────────────────

    private func currentBrowserURL() -> String? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = app.processIdentifier
        let bundle = app.bundleIdentifier ?? ""
        let appEl = AXUIElementCreateApplication(pid)

        if bundle == "com.apple.Safari" {
            return safariURL(appEl: appEl)
        }
        return genericBrowserURL(appEl: appEl)
    }

    private func safariURL(appEl: AXUIElement) -> String? {
        var windowVal: CFTypeRef?
        AXUIElementCopyAttributeValue(appEl, kAXMainWindowAttribute as CFString, &windowVal)
        guard let window = windowVal as! AXUIElement? else { return nil }

        var toolbarVal: CFTypeRef?
        AXUIElementCopyAttributeValue(window, "AXToolbar" as CFString, &toolbarVal)
        guard let toolbar = toolbarVal as! AXUIElement? else { return nil }

        var childrenVal: CFTypeRef?
        AXUIElementCopyAttributeValue(toolbar, kAXChildrenAttribute as CFString, &childrenVal)
        guard let children = childrenVal as? [AXUIElement] else { return nil }

        for child in children {
            if let url = findURLInSubtree(element: child, depth: 0, maxDepth: 4) {
                return url
            }
        }
        return nil
    }

    private func genericBrowserURL(appEl: AXUIElement) -> String? {
        var windowVal: CFTypeRef?
        AXUIElementCopyAttributeValue(appEl, kAXMainWindowAttribute as CFString, &windowVal)
        guard let window = windowVal as! AXUIElement? else { return nil }
        return findURLInSubtree(element: window, depth: 0, maxDepth: 6)
    }

    private func findURLInSubtree(element: AXUIElement, depth: Int, maxDepth: Int) -> String? {
        guard depth <= maxDepth else { return nil }

        var roleVal: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleVal)
        let role = roleVal as? String ?? ""

        if role == "AXTextField" {
            var val: CFTypeRef?
            AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &val)
            if let str = val as? String, str.hasPrefix("http") {
                return str
            }
        }

        var childrenVal: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenVal)
        guard let children = childrenVal as? [AXUIElement] else { return nil }

        for child in children {
            if let url = findURLInSubtree(element: child, depth: depth + 1, maxDepth: maxDepth) {
                return url
            }
        }
        return nil
    }

    private func currentBrowserTitle() -> String? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let appEl = AXUIElementCreateApplication(app.processIdentifier)

        var windowVal: CFTypeRef?
        AXUIElementCopyAttributeValue(appEl, kAXMainWindowAttribute as CFString, &windowVal)
        guard let window = windowVal as! AXUIElement? else { return nil }

        var titleVal: CFTypeRef?
        AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &titleVal)
        return titleVal as? String
    }

    private func isComposingMessage(element: AXUIElement) -> Bool {
        var roleVal: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleVal)
        guard let role = roleVal as? String else { return false }
        return role == "AXTextArea" || role == "AXTextField" || role == "AXComboBox"
    }
}