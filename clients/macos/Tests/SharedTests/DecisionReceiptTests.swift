import XCTest
@testable import Shared

/// The app's receipt verifier against the engine's vectors
/// (tests/vectors/trail_decision_v0, made by
/// scripts/build_trail_decision_vectors.py). The Python suite checks the
/// same files, so a JCS or signature difference between the two fails here.
final class DecisionReceiptTests: XCTestCase {
    private let vectors = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // SharedTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // macos
        .deletingLastPathComponent()  // clients
        .deletingLastPathComponent()  // repo root
        .appendingPathComponent("tests/vectors/trail_decision_v0")

    private func json(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: vectors.appendingPathComponent(name))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testEveryVectorGetsItsExpectedVerdict() throws {
        let expected = try json("expected.json")
        let hashes = try json("trail-hashes.json") as? [String: String] ?? [:]
        let pem = try String(contentsOf: vectors.appendingPathComponent("issuer-es256.pub.pem"),
                             encoding: .utf8)
        XCTAssertGreaterThanOrEqual(expected.count, 6)
        for (name, raw) in expected {
            let want = try XCTUnwrap(raw as? [String: Bool])
            let receipt = try DecisionReceipt(url: vectors.appendingPathComponent(name))
            let got = receipt.verify(publicKeyPEM: pem,
                                     trailRecordHash: hashes[receipt.recordId] ?? "")
            XCTAssertEqual(got.signature, want["signature"], "\(name) signature")
            XCTAssertEqual(got.evidence, want["evidence"], "\(name) evidence")
            XCTAssertEqual(got.trail, want["trail"], "\(name) trail")
        }
    }

    func testWrongKeyFails() throws {
        let receipt = try DecisionReceipt(url: vectors.appendingPathComponent("valid/0-allow.json"))
        let other = P256Key.otherPEM
        XCTAssertFalse(receipt.verify(publicKeyPEM: other, trailRecordHash: nil).signature)
    }

    func testJCSMatchesRFC8785Escapes() {
        let s = "q\"b\\n\n\t\u{01}/ä€😀\u{2028}"
        let value: [String: Any] = ["b": 1, "a": s, "A": [true, NSNull()] as [Any]]
        let out = String(data: JCS.encode(value)!, encoding: .utf8)
        XCTAssertEqual(out, "{\"A\":[true,null],\"a\":\"q\\\"b\\\\n\\n\\t\\u0001/ä€😀\u{2028}\",\"b\":1}")
    }

    func testFloatIsRejected() {
        let value: [String: Any] = ["x": 0.5]
        XCTAssertNil(JCS.encode(value))
    }
}

import CryptoKit
private enum P256Key {
    static let otherPEM = P256.Signing.PrivateKey().publicKey.pemRepresentation
}
