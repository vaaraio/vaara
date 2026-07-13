// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "VaaraMenuBar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "VaaraMenuBar",
            path: "Sources/VaaraMenuBar",
            resources: [.copy("Resources/icons")]
        )
    ]
)
