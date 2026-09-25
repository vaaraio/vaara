import CryptoKit
import Foundation

/// One signed decision receipt the engine wrote beside a trail
/// (`<trail dir>/receipts/<YYYY-MM-DD>/<record id>.json`, see
/// src/vaara/audit/decision_receipts.py), and its verification.
///
/// The app checks a receipt with its own code, not by asking the engine:
/// the ES256 signature over the JCS-canonical envelope, then
/// `sha256(JCS(evidence)) == evidenceRef.digest`, then the evidence's
/// `recordHash` against the record the trail holds. JCS here is the subset
/// the receipts use: objects, arrays, strings, integers, booleans and null.
/// Floats are banned on the wire, so a float in a receipt fails the check
/// rather than being canonicalized.
public struct DecisionReceipt: Identifiable {
    public let url: URL
    public let envelope: [String: Any]
    public let evidence: [String: Any]

    public var id: String { url.path }
    public var recordId: String { evidence["recordId"] as? String ?? "" }
    public var tool: String { evidence["toolName"] as? String ?? "" }
    public var agent: String { evidence["agentId"] as? String ?? "" }
    public var reason: String { evidence["reason"] as? String ?? "" }
    public var riskScore: String { evidence["riskScore"] as? String ?? "" }
    public var decidedAt: String { evidence["decidedAt"] as? String ?? "" }
    public var recordHash: String { evidence["recordHash"] as? String ?? "" }
    /// The envelope's verdict: allow, escalate or block.
    public var decision: String {
        (envelope["decisionDerived"] as? [String: Any])?["decision"] as? String ?? ""
    }
    public var keyId: String {
        (envelope["issuerAsserted"] as? [String: Any])?["secretVersion"] as? String ?? ""
    }

    public enum LoadError: Error { case notAReceipt }

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let env = obj["receipt"] as? [String: Any],
              let ev = obj["evidence"] as? [String: Any]
        else { throw LoadError.notAReceipt }
        self.url = url
        self.envelope = env
        self.evidence = ev
    }

    /// The file as written, for display and export.
    public var prettyJSON: String {
        (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }

    /// Verify against the issuer's public key (PEM) and, when the trail was
    /// readable, the hash the trail holds for this record (`nil` = no trail
    /// to check against; an empty string = the trail has no such record).
    public func verify(publicKeyPEM: String, trailRecordHash: String?) -> ReceiptCheck {
        let signed = ["version", "alg", "backLink", "decisionDerived", "issuerAsserted"]
        var body: [String: Any] = [:]
        for k in signed {
            guard let v = envelope[k] else {
                return ReceiptCheck(signature: false, evidence: false, trail: nil,
                                    detail: "envelope has no \(k)")
            }
            body[k] = v
        }
        guard envelope["alg"] as? String == "ES256" else {
            return ReceiptCheck(signature: false, evidence: false, trail: nil,
                                detail: "not an ES256 receipt")
        }

        var signatureOK = false
        var detail = ""
        if let payload = JCS.encode(body),
           let sigHex = envelope["signature"] as? String,
           let raw = Data(hex: sigHex), raw.count == 64,
           let key = try? P256.Signing.PublicKey(pemRepresentation: publicKeyPEM),
           let sig = try? P256.Signing.ECDSASignature(rawRepresentation: raw) {
            signatureOK = key.isValidSignature(sig, for: payload)
        }
        if !signatureOK { detail = "signature does not verify" }

        let ref = (envelope["decisionDerived"] as? [String: Any])?["evidenceRef"] as? [String: Any]
        var evidenceOK = false
        if let bytes = JCS.encode(evidence), let want = ref?["digest"] as? String {
            evidenceOK = want == "sha256:" + SHA256.hash(data: bytes).hex
        }
        if signatureOK && !evidenceOK { detail = "evidence digest does not match" }

        var trailOK: Bool?
        if let stored = trailRecordHash {
            if stored.isEmpty {
                trailOK = false
                if detail.isEmpty { detail = "record not in the trail" }
            } else {
                trailOK = "sha256:" + stored == recordHash
                if trailOK == false && detail.isEmpty { detail = "trail record hash differs" }
            }
        }
        return ReceiptCheck(signature: signatureOK, evidence: evidenceOK, trail: trailOK, detail: detail)
    }
}

public struct ReceiptCheck: Equatable {
    public let signature: Bool
    public let evidence: Bool
    /// nil when no trail was checked.
    public let trail: Bool?
    public let detail: String
    public var ok: Bool { signature && evidence && trail != false }

    public init(signature: Bool, evidence: Bool, trail: Bool?, detail: String) {
        self.signature = signature
        self.evidence = evidence
        self.trail = trail
        self.detail = detail
    }
}

/// RFC 8785 JSON Canonicalization over the value types a receipt can hold.
public enum JCS {
    public static func encode(_ value: Any) -> Data? {
        var out = ""
        guard write(value, into: &out) else { return nil }
        return Data(out.utf8)
    }

    private static func write(_ value: Any, into out: inout String) -> Bool {
        switch value {
        case is NSNull:
            out += "null"
        case let s as String:
            writeString(s, into: &out)
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                out += n.boolValue ? "true" : "false"
            } else {
                // Integers only: a float has no place on the wire.
                let type = String(cString: n.objCType)
                guard !["d", "f"].contains(type) else { return false }
                out += String(n.int64Value)
            }
        case let a as [Any]:
            out += "["
            for (i, v) in a.enumerated() {
                if i > 0 { out += "," }
                guard write(v, into: &out) else { return false }
            }
            out += "]"
        case let o as [String: Any]:
            // Keys sort by UTF-16 code units, as RFC 8785 section 3.2.3 says.
            let keys = o.keys.sorted { $0.utf16.lexicographicallyPrecedes($1.utf16) }
            out += "{"
            for (i, k) in keys.enumerated() {
                if i > 0 { out += "," }
                writeString(k, into: &out)
                out += ":"
                guard write(o[k]!, into: &out) else { return false }
            }
            out += "}"
        default:
            return false
        }
        return true
    }

    private static func writeString(_ s: String, into out: inout String) {
        out += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if u.value < 0x20 {
                    out += String(format: "\\u%04x", u.value)
                } else {
                    out.unicodeScalars.append(u)
                }
            }
        }
        out += "\""
    }
}

extension Data {
    init?(hex: String) {
        guard hex.count % 2 == 0 else { return nil }
        var bytes = [UInt8]()
        bytes.reserveCapacity(hex.count / 2)
        var idx = hex.startIndex
        while idx < hex.endIndex {
            let next = hex.index(idx, offsetBy: 2)
            guard let b = UInt8(hex[idx..<next], radix: 16) else { return nil }
            bytes.append(b)
            idx = next
        }
        self.init(bytes)
    }
}

extension Digest {
    var hex: String { map { String(format: "%02x", $0) }.joined() }
}
