# Draft formula for vaaraio/homebrew-tap. Not published yet: lands in the
# tap once the app is tested and its first tagged release exists.
#
# Builds from source on the user's machine, so the binary carries no
# quarantine attribute and Gatekeeper never prompts: no Developer ID
# needed. Requires the Xcode Command Line Tools (brew prompts if absent).
class VaaraApp < Formula
  desc "Vaara menu-bar app: the AI governance engine with a face"
  homepage "https://vaara.io"
  url "https://github.com/vaaraio/vaara/archive/refs/tags/vAPP_VERSION.tar.gz"
  sha256 "FILLED_ON_RELEASE"
  license "AGPL-3.0-or-later"

  depends_on :macos
  depends_on xcode: :build

  def install
    cd "clients/macos" do
      system "./build.sh"
      prefix.install "dist/Vaara.app"
    end
  end

  def caveats
    <<~EOS
      Put it in Applications and start it:
        cp -R #{prefix}/Vaara.app /Applications/ && open /Applications/Vaara.app
      Add it to System Settings > General > Login Items to start with the Mac.
      The engine is a separate install: brew install vaaraio/tap/vaara
      (or use the app's Setup tab, which detects and guides the install).
    EOS
  end

  test do
    assert_predicate prefix/"Vaara.app/Contents/MacOS/Vaara", :exist?
  end
end
