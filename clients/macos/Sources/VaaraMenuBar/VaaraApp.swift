// Menu-bar entry point: the Vaara mark tinted by the gate's state, with
// an optional live activity sparkline beside it. No numbers anywhere;
// the bar heights are the pulse, the colors are the verdicts.

import AppKit
import SwiftUI

@main
struct VaaraApp: App {
    @StateObject private var model = GateModel()
    @State private var approvals: ApprovalWindowManager?

    var body: some Scene {
        MenuBarExtra {
            ContentView(model: model)
                .onAppear {
                    model.start()
                    if approvals == nil {
                        approvals = ApprovalWindowManager(model: model)
                    }
                }
        } label: {
            Image(nsImage: menuImage(model))
        }
        .menuBarExtraStyle(.window)
    }

    private func stateColor(_ state: GateState) -> NSColor {
        switch state {
        case .green:  NSColor(red: 0.37, green: 0.72, blue: 0.47, alpha: 1)
        case .yellow: NSColor(red: 0.90, green: 0.76, blue: 0.29, alpha: 1)
        case .red:    NSColor(red: 0.85, green: 0.34, blue: 0.34, alpha: 1)
        }
    }

    private func markImage(for state: GateState) -> NSImage {
        if let url = Bundle.module.url(
            forResource: "icons/vaara-\(state.rawValue)", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            return img
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
    private func menuImage(_ model: GateModel) -> NSImage {
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
