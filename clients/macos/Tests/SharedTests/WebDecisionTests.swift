import XCTest
@testable import Shared

/// The WebKit filter refuses what `vaara check` could not decide.
///
/// The old parse allowed on a non-JSON answer, a crash, a missing binary and
/// a reply without `allowed`, which made a broken install indistinguishable
/// from a permissive policy. It also read an escalation as a deny.
final class WebDecisionTests: XCTestCase {
    private func json(_ object: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: object)
    }

    func testAnAllowIsAnAllow() {
        let d = WebDecision.parse(
            output: json(["allowed": true, "decision": "allow", "reason": "", "action_id": "a1"]),
            terminationStatus: 0)
        XCTAssertEqual(d, WebDecision(allowed: true, reason: nil, actionId: "a1"))
        XCTAssertFalse(d.engineUnavailable)
    }

    func testADenyCarriesItsReason() {
        let d = WebDecision.parse(
            output: json(["allowed": false, "decision": "deny", "reason": "credential host"]),
            terminationStatus: 1)
        XCTAssertFalse(d.allowed)
        XCTAssertEqual(d.reason, "credential host")
        XCTAssertFalse(d.engineUnavailable)
    }

    func testAnEscalationIsCarriedAsOne() {
        let d = WebDecision.parse(
            output: json(["allowed": false, "decision": "escalate", "reason": "borderline"]),
            terminationStatus: 1)
        XCTAssertFalse(d.allowed)
        XCTAssertEqual(d.reason, "escalate")
    }

    func testNonJSONOutputIsRefusedAsUnavailable() {
        let d = WebDecision.parse(output: "Traceback (most recent call last)".data(using: .utf8),
                                  terminationStatus: 1)
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable, d.reason ?? "")
    }

    func testEmptyOutputIsRefusedAsUnavailable() {
        let d = WebDecision.parse(output: Data(), terminationStatus: 0)
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable)
    }

    func testAnObjectWithoutAllowedIsRefusedAsUnavailable() {
        let d = WebDecision.parse(output: json(["decision": "allow"]), terminationStatus: 0)
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable)
        XCTAssertTrue(d.reason?.contains("allowed") ?? false)
    }

    func testAnAllowedThatIsNotABoolIsRefused() {
        let d = WebDecision.parse(output: json(["allowed": "true"]), terminationStatus: 0)
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable)
    }

    func testABinaryThatCouldNotRunIsRefusedAsUnavailable() {
        let d = WebDecision.parse(output: nil, terminationStatus: nil,
                                  launchError: "No such file or directory")
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable)
        XCTAssertTrue(d.reason?.contains("No such file") ?? false)
    }

    func testAProcessThatNeverRanIsRefused() {
        let d = WebDecision.parse(output: nil, terminationStatus: nil)
        XCTAssertFalse(d.allowed)
        XCTAssertTrue(d.engineUnavailable)
    }

    func testTheUnavailableReplyIsRecognisedByThePrefix() {
        let d = WebDecision.unavailable("vaara check timed out")
        XCTAssertEqual(d.reason, "unavailable: vaara check timed out")
        XCTAssertTrue(d.engineUnavailable)
        XCTAssertFalse(WebDecision(allowed: false, reason: "unavailable host list", actionId: nil)
            .engineUnavailable)
    }
}
