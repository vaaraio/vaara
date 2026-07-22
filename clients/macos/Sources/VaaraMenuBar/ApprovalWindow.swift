// The approval, made impossible to miss. A menu-bar app's popover only
// draws when the user clicks the icon, so an escalation could sit unseen
// behind a yellow mark. This raises the app's OWN floating window (not a
// macOS notification, which is permission-gated and easy to miss) the
// moment an action needs a human, centered and above everything, and
// tears it down when answered. No Notification Center, no permission.

import AppKit
import Combine
import SwiftUI

/// The standalone approval card, reused by both the popover and the
/// floating window. Self-contained: derives its own palette from the
/// model's appearance so it renders identically in either host.
struct ApprovalPanelView: View {
    @ObservedObject var model: GateModel
    let pending: PendingApproval

    private var dark: Bool { model.config.appearance != "light" }
    private var p: Palette { dark ? .dark : .light }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Circle()
                    .fill(GateState.yellow.color)
                    .frame(width: 11, height: 11)
                    .shadow(color: GateState.yellow.color.opacity(0.6), radius: 7)
                Text("Approval needed")
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(p.ink)
                Spacer()
            }
            .padding(.horizontal, 22).padding(.top, 24).padding(.bottom, 16)

            Rectangle().fill(p.hairline).frame(height: 1)

            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("AN AI AGENT WANTS TO")
                        .font(.system(size: 9, weight: .medium)).tracking(1.2)
                        .foregroundStyle(p.ghost)
                    Text(pending.toolName)
                        .font(.system(size: 15, design: .monospaced))
                        .foregroundStyle(p.ink)
                        .lineLimit(2).truncationMode(.middle)
                }
                if !pending.reason.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("WHY IT WAS FLAGGED")
                            .font(.system(size: 9, weight: .medium)).tracking(1.2)
                            .foregroundStyle(p.ghost)
                        Text(pending.reason)
                            .font(.system(size: 12))
                            .foregroundStyle(p.faint)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.horizontal, 22).padding(.vertical, 18)

            Rectangle().fill(p.hairline).frame(height: 1)

            HStack(spacing: 12) {
                Button {
                    model.resolveApproval(pending.actionID, approve: false)
                } label: {
                    Text("Block")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity).padding(.vertical, 10)
                        .background(GateState.red.color)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain).keyboardShortcut(.cancelAction)

                Button {
                    model.resolveApproval(pending.actionID, approve: true)
                } label: {
                    Text("Approve")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(dark ? .black : .white)
                        .frame(maxWidth: .infinity).padding(.vertical, 10)
                        .background(GateState.green.color)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain).keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 22).padding(.vertical, 16)

            Text("Doing nothing blocks the action when the request times out.")
                .font(.system(size: 10))
                .foregroundStyle(p.ghost)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.bottom, 16)
        }
        .frame(width: 400)
        .background(p.wash)
        .background(.ultraThinMaterial)
        .environment(\.colorScheme, dark ? .dark : .light)
    }
}

/// The notch, lowered. A compact black card matched to the notch width,
/// square-ish at the top (flush, as if the cutout detached) and rounded
/// wider at the bottom (the cutout flares wider there). It carries just
/// what a person needs to answer: what the agent wants, and two buttons.
/// The window slides it down; nothing expands. Always black so it reads
/// as the notch itself, independent of the app's light/dark theme.
/// The card outline that fuses to the notch. The two top corners are
/// CONCAVE (scooped, centered on the outer corner) so they interlock with
/// the notch's own rounded bottom corners instead of leaving triangular
/// slivers. The bottom corners are ordinary convex rounds. Off notch,
/// pass topConcave = 0 and topConvex > 0 for a normal rounded top.
struct NotchFusedShape: Shape {
    var topConcave: CGFloat = 0
    var topConvex: CGFloat = 0
    var bottom: CGFloat = 22

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let w = rect.width, h = rect.height, b = bottom
        let tc = topConcave

        if tc > 0 {
            // Concave top corners (scooped toward the outer corner point).
            p.move(to: CGPoint(x: tc, y: 0))
            p.addLine(to: CGPoint(x: w - tc, y: 0))
            p.addArc(center: CGPoint(x: w, y: 0), radius: tc,
                     startAngle: .degrees(180), endAngle: .degrees(90),
                     clockwise: true)
            p.addLine(to: CGPoint(x: w, y: h - b))
            p.addArc(center: CGPoint(x: w - b, y: h - b), radius: b,
                     startAngle: .degrees(0), endAngle: .degrees(90),
                     clockwise: false)
            p.addLine(to: CGPoint(x: b, y: h))
            p.addArc(center: CGPoint(x: b, y: h - b), radius: b,
                     startAngle: .degrees(90), endAngle: .degrees(180),
                     clockwise: false)
            p.addLine(to: CGPoint(x: 0, y: tc))
            p.addArc(center: CGPoint(x: 0, y: 0), radius: tc,
                     startAngle: .degrees(90), endAngle: .degrees(0),
                     clockwise: true)
            p.closeSubpath()
        } else {
            let t = topConvex
            p.move(to: CGPoint(x: t, y: 0))
            p.addLine(to: CGPoint(x: w - t, y: 0))
            p.addArc(center: CGPoint(x: w - t, y: t), radius: t,
                     startAngle: .degrees(-90), endAngle: .degrees(0), clockwise: false)
            p.addLine(to: CGPoint(x: w, y: h - b))
            p.addArc(center: CGPoint(x: w - b, y: h - b), radius: b,
                     startAngle: .degrees(0), endAngle: .degrees(90), clockwise: false)
            p.addLine(to: CGPoint(x: b, y: h))
            p.addArc(center: CGPoint(x: b, y: h - b), radius: b,
                     startAngle: .degrees(90), endAngle: .degrees(180), clockwise: false)
            p.addLine(to: CGPoint(x: 0, y: t))
            p.addArc(center: CGPoint(x: t, y: t), radius: t,
                     startAngle: .degrees(180), endAngle: .degrees(270), clockwise: false)
            p.closeSubpath()
        }
        return p
    }
}

struct NotchApprovalView: View {
    let pending: PendingApproval
    let onDecision: (Bool) -> Void
    let width: CGFloat
    var topRadius: CGFloat = 0    // convex top round, off-notch only
    var notchCorner: CGFloat = 0  // concave top corners that hug the notch

    private let ink = Color(red: 0.90, green: 0.92, blue: 0.91)
    private let faint = Color(red: 0.60, green: 0.64, blue: 0.62)
    private let green = GateState.green.color
    private let red = GateState.red.color

    var body: some View {
        VStack(spacing: 9) {
            HStack(spacing: 7) {
                Circle().fill(GateState.yellow.color).frame(width: 7, height: 7)
                Text("Approval needed")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(ink)
                Spacer(minLength: 0)
            }
            Text(pending.toolName)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(faint)
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            HStack(spacing: 8) {
                Button { onDecision(false) } label: {
                    Text("Block")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity).padding(.vertical, 7)
                        .background(red).clipShape(RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain).keyboardShortcut(.cancelAction)
                Button { onDecision(true) } label: {
                    Text("Approve")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity).padding(.vertical, 7)
                        .background(green).clipShape(RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain).keyboardShortcut(.defaultAction)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 9)
        .padding(.bottom, 11)
        .frame(width: width)
        // Pure #000000, no shadow, no stroke, top corners square so the
        // top edge fuses with the physical notch: it reads as the notch
        // itself lowering. Only the bottom corners round.
        .background(Color.black)
        .clipShape(NotchFusedShape(
            topConcave: notchCorner, topConvex: topRadius, bottom: 22))
        .environment(\.colorScheme, .dark)
    }
}

/// Watches the model's pendingApproval and raises/dismisses a floating
/// window so an escalation surfaces without the user opening the menu.
@MainActor
final class ApprovalWindowManager {
    private let model: GateModel
    private var window: NSPanel?
    private var cancellable: AnyCancellable?

    init(model: GateModel) {
        self.model = model
        cancellable = model.$pendingApproval.sink { [weak self] pending in
            Task { @MainActor in self?.update(pending) }
        }
    }

    private func update(_ pending: PendingApproval?) {
        if let pending {
            show(pending)
        } else {
            window?.close()
            window = nil
        }
    }

    /// Notch geometry from the screen. Width is the gap between the two
    /// auxiliary top areas; height is the safe-area top inset. Zero when
    /// the Mac has no notch (safeAreaInsets.top == 0).
    private func notchSize(_ screen: NSScreen) -> CGSize {
        let h = screen.safeAreaInsets.top
        guard h > 0 else { return .zero }
        var w: CGFloat = 200
        if let left = screen.auxiliaryTopLeftArea,
           let right = screen.auxiliaryTopRightArea {
            w = screen.frame.width - left.width - right.width
        }
        return CGSize(width: w, height: h)
    }

    private func show(_ pending: PendingApproval) {
        let panel = window ?? makePanel()
        window = panel

        guard let screen = NSScreen.main else {
            let host = NSHostingView(rootView: NotchApprovalView(
                pending: pending,
                onDecision: { [weak self] ok in
                    self?.model.resolveApproval(pending.actionID, approve: ok) },
                width: 260, topRadius: 20))
            panel.contentView = host
            panel.setContentSize(host.fittingSize)
            panel.center(); panel.orderFrontRegardless(); return
        }

        // Card width hugs the notch, floored just enough that the two
        // buttons stay readable (and used as-is on no-notch displays).
        // It slides down as a unit; it does not expand.
        let notch = notchSize(screen)
        let hasNotch = notch.height > 0
        let cardWidth = hasNotch ? max(notch.width, 200) : 220

        let host = NSHostingView(rootView: NotchApprovalView(
            pending: pending,
            onDecision: { [weak self] ok in
                self?.model.resolveApproval(pending.actionID, approve: ok) },
            width: cardWidth,
            topRadius: hasNotch ? 0 : 20,
            notchCorner: hasNotch ? 10 : 0))
        panel.contentView = host
        let size = host.fittingSize

        let full = screen.frame
        let x = full.midX - size.width / 2
        // Start with the top edge at the very top of the screen (the card
        // tucked in the notch), then lower it into place.
        let startY = full.maxY - size.height
        // Rest so the card's TOP edge sits at the notch's bottom edge, with
        // a 2pt overlap up into it to kill any seam. The menu bar is a touch
        // taller than the notch, so resting at visibleFrame.maxY left a
        // small gap; anchoring to the notch height itself closes it. Off
        // notch, rest just below the menu bar instead.
        let restTop = hasNotch ? full.maxY - notch.height + 2 : screen.visibleFrame.maxY
        let restY = restTop - size.height

        panel.setFrame(NSRect(x: x, y: startY, width: size.width, height: size.height),
                       display: true)
        panel.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)

        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.34
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().setFrame(
                NSRect(x: x, y: restY, width: size.width, height: size.height),
                display: true)
        }
    }

    private func makePanel() -> NSPanel {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 320),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false        // pure black card, no window chrome
        panel.isMovableByWindowBackground = true
        panel.level = .statusBar                // above menu-bar apps' windows
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        return panel
    }
}
