import XCTest
@testable import Shared

/// The app's approval MAC against the engine's vectors
/// (tests/fixtures/approval_v1). The Python suite checks the same file, so a
/// difference in the message layout or the key encoding fails here.
final class ApprovalSigningTests: XCTestCase {
    private let vectors = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // SharedTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // macos
        .deletingLastPathComponent()  // clients
        .deletingLastPathComponent()  // repo root
        .appendingPathComponent("tests/fixtures/approval_v1/vectors.json")

    func testEveryVectorMAC() throws {
        let data = try Data(contentsOf: vectors)
        let doc = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let key = try XCTUnwrap(ApprovalSigning.hexData(try XCTUnwrap(doc["key_hex"] as? String)))
        let cases = try XCTUnwrap(doc["cases"] as? [[String: String]])
        XCTAssertGreaterThanOrEqual(cases.count, 4)
        for c in cases {
            let got = ApprovalSigning.mac(key: key, actionID: c["action_id"]!,
                                          nonce: c["nonce"]!, decision: c["decision"]!)
            XCTAssertEqual(got, c["mac"], "\(c["action_id"]!) \(c["decision"]!)")
        }
    }

    func testKeyLivesInKeysBesideTheApprovalsDir() {
        let dir = URL(fileURLWithPath: "/Users/x/.vaara/approvals")
        XCTAssertEqual(ApprovalSigning.keyURL(approvalsDir: dir).path,
                       "/Users/x/.vaara/keys/approval-hmac.key")
    }

    func testBadHexIsRejected() {
        XCTAssertNil(ApprovalSigning.hexData("abc"))
        XCTAssertNil(ApprovalSigning.hexData("zz"))
        XCTAssertNil(ApprovalSigning.hexData(""))
    }
}
