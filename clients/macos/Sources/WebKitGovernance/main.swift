import Foundation
import NetworkExtension

// A NetworkExtension system extension is a standalone executable, not an
// appex, so it needs its own entry point. Without this the target compiles
// and then fails to link with "Undefined symbols: _main".
//
// startSystemExtensionMode reads NEProviderClasses from Info.plist and
// instantiates the provider named there. dispatchMain parks the main thread
// so the process stays alive to service flows.
autoreleasepool {
    NEProvider.startSystemExtensionMode()
}

dispatchMain()
