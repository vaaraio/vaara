// Menu-bar entry point: the Vaara mark tinted by the gate's state, with
// an optional live activity sparkline beside it. No numbers anywhere;
// the bar heights are the pulse, the colors are the verdicts.

import SwiftUI

@main
struct VaaraApp: App {
    @StateObject private var model = GateModel()
    @State private var approvals: ApprovalWindowManager?
    /// Startup runs from two places (the menu bar label at launch, the
    /// popover content on first open) and must happen once.
    @State private var started = false

    // XPC service the network filter extension connects to.
    private let policyService = PolicyServiceDelegate()
    // Shared instance: AccessibilityObserver vends `.shared` and holds its own
    // observer table, so constructing a second one left two observers running.
    private let accessibilityObserver = AccessibilityObserver.shared
    @StateObject private var systemExtension = SystemExtensionManager.shared

    var body: some Scene {
        MenuBarExtra {
            ContentView(model: model)
                .onAppear { startEverything() }
        } label: {
            Image(nsImage: menuImage())
                // The label renders as soon as the menu bar item exists, so
                // this is the launch path. Hanging startup off the content's
                // onAppear alone meant nothing ran until the user clicked the
                // icon: no trail polling, no XPC listener for the filter, no
                // Accessibility observer and no approval window. A governance
                // app that only governs while its popover is open is not
                // governing. Both call sites go through the same guarded
                // helper, so opening the popover afterwards is a no-op.
                .onAppear { startEverything() }
        }
        .menuBarExtraStyle(.window)
    }

    /// Bring up every background piece exactly once.
    private func startEverything() {
        guard !started else { return }
        started = true
        model.start()
        // Start the XPC listener the filter extension connects to.
        _ = policyService
        // Start Accessibility observer for UI context.
        accessibilityObserver.start()
        // Report whether the filter extension is installed and enabled.
        // Activation itself is user-initiated from Setup; macOS prompts for
        // approval.
        systemExtension.refresh()
        if approvals == nil {
            approvals = ApprovalWindowManager(model: model)
        }
    }

    private func stateColor(_ state: GateState) -> NSColor {
        switch state {
        case .green:  NSColor(red: 0.37, green: 0.72, blue: 0.47, alpha: 1)
        case .yellow: NSColor(red: 0.90, green: 0.76, blue: 0.29, alpha: 1)
        case .red:    NSColor(red: 0.85, green: 0.34, blue: 0.34, alpha: 1)
        }
    }

    private func markImage(for state: GateState) -> NSImage {
        let name = "vaara-\(state.rawValue)"

        // Xcode collapses a foo.png / foo@2x.png pair into a single
        // foo.tiff written to the top of Contents/Resources, so the
        // icons/ subdirectory and the .png names the file lookup below
        // expects do not exist in an Xcode-built bundle. That lookup failed
        // silently and the mark fell through to the plain coloured oval,
        // which is why an Xcode build showed a green ball where the Homebrew
        // build, which copies icons/ verbatim, showed the Vaara mark.
        // Asking the bundle by name finds the tiff.
        if let img = Bundle.main.image(forResource: name) {
            return img
        }

        let bundle = Bundle.main
        let resourcePath = bundle.resourcePath ?? ""
        for candidate in ["\(resourcePath)/icons/\(name).png",
                          "\(resourcePath)/\(name).png"] {
            if let img = NSImage(contentsOfFile: candidate) {
                return img
            }
        }

        let color = stateColor(state)
        let img = NSImage(size: NSSize(width: 18, height: 18), flipped: false) { rect in
            color.setFill()
            NSBezierPath(ovalIn: rect.insetBy(dx: 3, dy: 3)).fill()
            return true
        }
        return img
    }

    /// The full label: mark, plus (when enabled) a 12-bar sparkline of
    /// the last two minutes of activity.
    private func menuImage() -> NSImage {
        let mark = markImage(for: model.state)
        let graphOn = model.config.menubar_graph
        let buckets = model.buckets

        let markSize: CGFloat = 18
        let barWidth: CGFloat = 2.5
        let barGap: CGFloat = 1.0
        let graphWidth: CGFloat = graphOn
            ? CGFloat(buckets.count) * (barWidth + barGap) + 5 : 0
        let width = markSize + graphWidth
        let height: CGFloat = 18

        let img = NSImage(size: NSSize(width: width, height: height), flipped: false) { _ in
            mark.draw(in: NSRect(x: 0, y: 0, width: markSize, height: markSize))
            guard graphOn, !buckets.isEmpty else { return true }
            let maxCount = max(buckets.map(\.0).max() ?? 1, 1)
            var x = markSize + 5
            for (count, worst) in buckets {
                let floorH: CGFloat = 1.5          // an empty bucket still shows a tick
                let h = count == 0 ? floorH
                    : floorH + (height - 4 - floorH) * CGFloat(count) / CGFloat(maxCount)
                let color = self.stateColor(worst)
                color.withAlphaComponent(count == 0 ? 0.30 : 0.95).setFill()
                NSBezierPath(
                    roundedRect: NSRect(x: x, y: 2, width: barWidth, height: h),
                    xRadius: 1, yRadius: 1
                ).fill()
                x += barWidth + barGap
            }
            return true
        }
        img.isTemplate = false  // the colors ARE the signal
        return img
    }
}
