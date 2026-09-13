"""`vaara llm-proxy` must reach every option the real parser defines.

2026-09-13: the subcommand restated all fourteen options in cli.py and
re-serialised them for vaara.integrations.llm_proxy, so an option added there
and not mirrored was silently unreachable. Three were: --seal-file,
--seal-listen-unix and --allow-origin, the last being the origin guard that
stops a visited page from spending the upstream key. This pins the property
rather than the three names, so the next added option cannot drift either.
"""
import argparse
import io
import contextlib
import pytest

from vaara.integrations import llm_proxy as real
from vaara import cli


def _real_option_strings():
    """Every --option that vaara/integrations/llm_proxy.py declares.

    Read from the module source, because that file is the source of truth for
    what the command accepts. Introspecting a live parser would need the module
    to hand one back, and it builds its parser inside main().
    """
    import inspect
    import re

    src = inspect.getsource(real)
    found = set(re.findall(r'add_argument\(\s*["\'](--[a-z0-9-]+)["\']', src))
    return found - {"--help", "--version"}


def _cli_help_text():
    buf = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.main(["llm-proxy", "--help"])
    return buf.getvalue()


def test_every_real_option_is_reachable_through_the_vaara_command():
    text = _cli_help_text()
    assert text.strip(), "vaara llm-proxy --help printed nothing"
    missing = sorted(o for o in _real_option_strings() if o not in text)
    assert not missing, (
        "options defined in vaara.integrations.llm_proxy but unreachable "
        f"through `vaara llm-proxy`: {missing}"
    )


@pytest.mark.parametrize("flag", ["--seal-file", "--seal-listen-unix",
                                  "--allow-origin"])
def test_the_three_that_drifted(flag):
    assert flag in _cli_help_text()


def test_unknown_option_is_still_rejected():
    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.main(["llm-proxy", "--upstream", "https://x",
                      "--api-key", "k", "--bogus"])
    assert "--bogus" in buf.getvalue()


def test_missing_required_option_is_still_rejected():
    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.main(["llm-proxy"])
    assert "--upstream" in buf.getvalue()
