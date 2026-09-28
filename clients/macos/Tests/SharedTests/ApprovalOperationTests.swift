import XCTest
@testable import Shared

/// The approval card shows the operation the engine sent, not a summary.
final class ApprovalOperationTests: XCTestCase {
    func testShellCallReadsAsItsCommand() {
        let params: [String: Any] = ["command": "rm -rf build/", "description": "clean"]
        XCTAssertEqual(ApprovalOperation.render(params), "rm -rf build/")
    }

    func testOtherCallsShowEveryArgument() {
        let params: [String: Any] = ["table": "users", "query": "DELETE FROM users"]
        let text = ApprovalOperation.render(params)
        XCTAssertTrue(text.contains("\"query\" : \"DELETE FROM users\""), text)
        XCTAssertTrue(text.contains("\"table\" : \"users\""), text)
    }

    func testNoParametersIsEmpty() {
        XCTAssertEqual(ApprovalOperation.render(nil), "")
        XCTAssertEqual(ApprovalOperation.render(NSNull()), "")
    }
}
