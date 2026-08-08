// swift-tools-version:5.9
import PackageDescription

// Three build paths exist for this client and they must agree on which
// sources belong to which target:
//
//   * project.yml (XcodeGen -> xcodebuild) — what CI builds and what
//     produces the signed app plus the system extension.
//   * this file (SwiftPM) — what `swift build` uses.
//   * the Homebrew formula in vaaraio/homebrew-tap — raw swiftc.
//
// Sources/Shared holds the XPC protocol, the policy client and the
// governed-host list, and BOTH product targets reference those symbols.
// XcodeGen compiles Sources/Shared into each target directly; SwiftPM
// cannot share files between targets, so here it is a real module that
// both targets depend on. Consumers guard the import with
// `#if canImport(Shared)` so the same sources still compile under the
// XcodeGen and swiftc paths, where Shared is not a separate module.

let package = Package(
    name: "VaaraMenuBar",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "VaaraMenuBar", targets: ["VaaraMenuBar"]),
    ],
    targets: [
        .target(
            name: "Shared",
            path: "Sources/Shared"
        ),
        .executableTarget(
            name: "VaaraMenuBar",
            dependencies: ["Shared"],
            path: "Sources/VaaraMenuBar",
            resources: [.copy("Resources/icons")]
        ),
        .target(
            name: "WebKitGovernance",
            dependencies: ["Shared"],
            path: "Sources/WebKitGovernance",
            linkerSettings: [
                .linkedFramework("NetworkExtension"),
                .linkedFramework("OSLog"),
            ]
        ),
    ]
)
