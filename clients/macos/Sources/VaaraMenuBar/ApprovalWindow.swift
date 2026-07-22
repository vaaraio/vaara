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
        // Top edge flush to the notch, bottom corners rounded: the card
        // reads as lowering out of the menu bar itself.
        .clipShape(.rect(bottomLeadingRadius: 18, bottomTrailingRadius: 18))
        .overlay(
            UnevenRoundedRectangle(bottomLeadingRadius: 18, bottomTrailingRadius: 18)
                .stroke(p.hairline, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.35), radius: 18, y: 8)
        .environment(\.colorScheme, dark ? .dark : .light)
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

    private func show(_ pending: PendingApproval) {
        let host = NSHostingView(rootView: ApprovalPanelView(model: model, pending: pending))
        let size = host.fittingSize

        let panel = window ?? makePanel()
        window = panel
        panel.contentView = host

        guard let screen = NSScreen.main else {
            panel.setContentSize(size); panel.center()
            panel.orderFrontRegardless(); return
        }
        // Centered horizontally under the notch; rest just below the menu
        // bar. Start tucked up behind the bar, then slide down into place.
        let full = screen.frame
        let x = full.midX - size.width / 2
        let restY = screen.visibleFrame.maxY - size.height   // just under menu bar
        let startY = full.maxY - size.height                 // flush to top, hidden

        panel.setFrame(NSRect(x: x, y: startY, width: size.width, height: size.height),
                       display: true)
        panel.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)

        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.32
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
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.level = .statusBar                // above menu-bar apps' windows
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        return panel
    }
}
