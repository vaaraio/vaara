import Foundation

/// What an escalated call will do, as the human should read it before
/// approving. The engine puts the call's full arguments into the approval
/// request as `parameters`, with `parameters_sha256` over their canonical
/// JSON (src/vaara/approvals.py). A shell call reads as its command; any
/// other call reads as its arguments, pretty-printed with sorted keys, so
/// nothing the call carries is summarised away.
public enum ApprovalOperation {
    public static func render(_ parameters: Any?) -> String {
        guard let parameters, !(parameters is NSNull) else { return "" }
        if let text = parameters as? String { return text }
        if let dict = parameters as? [String: Any],
           let command = dict["command"] as? String, !command.isEmpty {
            return command
        }
        guard JSONSerialization.isValidJSONObject(parameters),
              let data = try? JSONSerialization.data(
                  withJSONObject: parameters,
                  options: [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]),
              let text = String(data: data, encoding: .utf8)
        else { return String(describing: parameters) }
        return text
    }
}
