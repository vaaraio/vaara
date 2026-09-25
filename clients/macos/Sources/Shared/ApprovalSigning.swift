import CryptoKit
import Foundation

/// Signs a human's answer to an escalated action, as the engine checks it
/// (src/vaara/approvals.py). The gate ignores a decision file without a valid
/// `mac`, because any process of the user can write a file into the
/// approvals directory; only one that can read the approval key can sign.
///
/// mac = hex HMAC-SHA256 under the key in `keys/approval-hmac.key` beside the
/// approvals directory (hex text, made by the gate on first use), over
/// `vaara-approval/v1`, the action id, the request's nonce and the decision,
/// one per line.
public enum ApprovalSigning {
    public static let context = "vaara-approval/v1"
    public static let keyName = "approval-hmac.key"

    public static func keyURL(approvalsDir: URL) -> URL {
        approvalsDir.deletingLastPathComponent()
            .appendingPathComponent("keys")
            .appendingPathComponent(keyName)
    }

    public static func loadKey(approvalsDir: URL) -> Data? {
        guard let text = try? String(contentsOf: keyURL(approvalsDir: approvalsDir), encoding: .utf8)
        else { return nil }
        return hexData(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    public static func mac(key: Data, actionID: String, nonce: String, decision: String) -> String {
        let message = [context, actionID, nonce, decision].joined(separator: "\n")
        let code = HMAC<SHA256>.authenticationCode(for: Data(message.utf8), using: SymmetricKey(data: key))
        return code.map { String(format: "%02x", $0) }.joined()
    }

    static func hexData(_ hex: String) -> Data? {
        guard hex.count % 2 == 0, !hex.isEmpty else { return nil }
        var out = Data(capacity: hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
            out.append(byte)
            index = next
        }
        return out
    }
}
