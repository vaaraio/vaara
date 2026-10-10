"""AMD SEV-SNP key chain, live report emission, and the attested enforcement tier.

The chain check runs on a real report from a Milan guest and the VCEK AMD's KDS
issued for it (tests/fixtures/sev_snp_milan). The negative cases build their own
ARK, ASK and VCEK, which can never chain to AMD's pinned root.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import struct
import uuid
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from vaara.attestation import _sev_snp_chain as chain_mod  # noqa: E402
from vaara.attestation._enforcement import (  # noqa: E402
    bind_record_to_report_data,
    verify_enforcement,
)
from vaara.attestation._enforcement_set import check_enforcement_set  # noqa: E402
from vaara.attestation._sev_snp_chain import (  # noqa: E402
    SNP_CERT_GUIDS,
    certificate_product,
    decode_tcb,
    parse_certificate_table,
    report_signer,
    split_pem_chain,
    verify_sev_snp_chain,
)
from vaara.attestation.tee import (  # noqa: E402
    MockSEVSNPAttester,
    SEVSNPHostAttester,
    TEEAttestationError,
    parse_sev_snp_report,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "sev_snp_milan"
MEASUREMENT = bytes(range(1, 49))


def _fix(name: str) -> bytes:
    return (FIX / name).read_bytes()


def _real_report():
    return parse_sev_snp_report(_fix("report.bin"))


def _vcek_pem(cert_bytes: bytes) -> bytes:
    cert = chain_mod._load_cert(cert_bytes)
    return cert.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


# ---- real AMD hardware --------------------------------------------------------


def test_real_milan_report_chains_to_amd_root():
    v = verify_sev_snp_chain(
        _real_report(), _fix("vcek.der"), _fix("ask.pem"), _fix("ark.pem"),
        at=dt.datetime(2026, 9, 28, tzinfo=dt.timezone.utc),
    )
    assert v.ok, v.reason
    assert v.product == "Milan" and v.signer == "vcek"
    assert all(
        getattr(v, f) for f in (
            "ark_pinned", "ark_self_signed", "intermediate_signed_by_ark",
            "key_signed_by_intermediate", "certificates_current", "hwid_match",
            "tcb_match", "report_signature_valid",
        )
    )


def test_real_report_fields_decode():
    report = _real_report()
    assert report_signer(report) == "vcek"
    assert decode_tcb(report.reported_tcb, "Milan") == {
        "bl": 2, "tee": 0, "snp": 5, "ucode": 68,
    }
    assert certificate_product(_fix("vcek.der")) == "Milan"
    assert certificate_product(_fix("ark.pem")) == "Milan"
    assert certificate_product(_fix("genoa_ark.pem")) == "Genoa"


def test_another_products_root_fails_the_pin():
    v = verify_sev_snp_chain(
        _real_report(), _fix("vcek.der"), _fix("genoa_ask.pem"),
        _fix("genoa_ark.pem"),
    )
    assert not v.ok and not v.ark_pinned
    assert not v.key_signed_by_intermediate
    assert "pinned Milan root" in v.reason


def test_tampered_report_fails_only_the_report_signature():
    raw = bytearray(_fix("report.bin"))
    raw[0x60] ^= 0x01  # inside REPORT_DATA
    v = verify_sev_snp_chain(
        parse_sev_snp_report(bytes(raw)), _fix("vcek.der"), _fix("ask.pem"),
        _fix("ark.pem"),
    )
    assert v.ark_pinned and v.key_signed_by_intermediate and v.tcb_match
    assert not v.report_signature_valid and not v.ok


def test_expired_certificates_fail():
    v = verify_sev_snp_chain(
        _real_report(), _fix("vcek.der"), _fix("ask.pem"), _fix("ark.pem"),
        at=dt.datetime(2040, 1, 1, tzinfo=dt.timezone.utc),
    )
    assert not v.certificates_current and not v.ok


def test_real_report_through_verify_enforcement():
    record = {"version": 1, "note": "not the record this report was made for"}
    v = verify_enforcement(
        record, _fix("report.bin"), _vcek_pem(_fix("vcek.der")),
        amd_chain=(_fix("vcek.der"), _fix("ask.pem"), _fix("ark.pem")),
    )
    assert v.signature_valid and v.vcek_chain_basis == "kds_verified"
    assert v.amd_chain is not None and v.amd_chain["ok"]
    # The report binds to some other record, so the verdict still fails.
    assert not v.bound and v.tier == "unverified" and not v.ok


def test_chain_key_must_be_the_key_that_checked_the_signature():
    other = ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    v = verify_enforcement(
        {"version": 1}, _fix("report.bin"), other,
        amd_chain=(_fix("vcek.der"), _fix("ask.pem"), _fix("ark.pem")),
    )
    assert v.vcek_chain_basis == "chain_failed"
    assert "not the key in vcek_pem" in v.reason


# ---- a self-made chain: never AMD, exercises every other check ----------------


def _name(cn: str) -> x509.Name:
    return x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


def _ext(oid: str, value: bytes) -> x509.UnrecognizedExtension:
    return x509.UnrecognizedExtension(x509.ObjectIdentifier(oid), value)


def _pss() -> padding.PSS:
    return padding.PSS(mgf=padding.MGF1(hashes.SHA384()), salt_length=48)


def _build(subject, issuer, key, signer, extensions=()):
    now = dt.datetime.now(dt.timezone.utc)
    b = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=365))
    )
    for ext in extensions:
        b = b.add_extension(ext, critical=False)
    cert = b.sign(signer, hashes.SHA384(), rsa_padding=_pss())
    return cert.public_bytes(serialization.Encoding.PEM)


def _spl(value: int) -> bytes:
    return bytes([0x02, 0x01, value])


@pytest.fixture(scope="module")
def mock_chain():
    """(vcek private key, ark PEM, ask PEM, make_vcek(product, hwid, spls))."""
    ark_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ask_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    vcek_key = ec.generate_private_key(ec.SECP384R1())
    ark = _build(_name("ARK-Milan"), _name("ARK-Milan"), ark_key.public_key(), ark_key)
    ask = _build(_name("SEV-Milan"), _name("ARK-Milan"), ask_key.public_key(), ark_key)

    def make_vcek(product="Milan-B0", hwid=bytes(64), bl=0, tee=0, snp=0, ucode=0):
        exts = [
            _ext("1.3.6.1.4.1.3704.1.2", bytes([0x16, len(product)]) + product.encode()),
            _ext("1.3.6.1.4.1.3704.1.3.1", _spl(bl)),
            _ext("1.3.6.1.4.1.3704.1.3.2", _spl(tee)),
            _ext("1.3.6.1.4.1.3704.1.3.3", _spl(snp)),
            _ext("1.3.6.1.4.1.3704.1.3.8", _spl(ucode)),
        ]
        if hwid is not None:
            exts.append(_ext("1.3.6.1.4.1.3704.1.4", hwid))
        return _build(
            _name("SEV-VCEK"), _name("SEV-Milan"), vcek_key.public_key(), ask_key, exts
        )

    return vcek_key, ark, ask, make_vcek


def _mock_report(key, report_data=bytes(64)) -> bytes:
    return MockSEVSNPAttester(key, measurement=MEASUREMENT).emit(report_data)


def test_self_made_root_is_never_amd(mock_chain):
    key, ark, ask, make_vcek = mock_chain
    report = parse_sev_snp_report(_mock_report(key))
    v = verify_sev_snp_chain(report, make_vcek(), ask, ark)
    # Every signature and match holds; only the pin separates it from AMD.
    assert v.ark_self_signed and v.intermediate_signed_by_ark
    assert v.key_signed_by_intermediate and v.hwid_match and v.tcb_match
    assert v.report_signature_valid
    assert not v.ark_pinned and not v.ok


@pytest.fixture
def pin_mock_root(mock_chain, monkeypatch):
    _, ark, _, _ = mock_chain
    digest = chain_mod._spki_sha256(chain_mod._load_cert(ark))
    monkeypatch.setitem(chain_mod.AMD_ARK_SPKI_SHA256, "Milan", digest)


def test_pinned_mock_root_passes(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    v = verify_sev_snp_chain(parse_sev_snp_report(_mock_report(key)), make_vcek(), ask, ark)
    assert v.ok, v.reason


def test_tcb_mismatch_fails(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    v = verify_sev_snp_chain(
        parse_sev_snp_report(_mock_report(key)), make_vcek(ucode=9), ask, ark
    )
    assert not v.tcb_match and not v.ok and "REPORTED_TCB" in v.reason


def test_hwid_mismatch_fails(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    v = verify_sev_snp_chain(
        parse_sev_snp_report(_mock_report(key)), make_vcek(hwid=b"\x01" * 64), ask, ark
    )
    assert not v.hwid_match and not v.ok and "CHIP_ID" in v.reason


def test_turin_hwid_is_eight_bytes(mock_chain):
    key, ark, ask, make_vcek = mock_chain
    report = parse_sev_snp_report(_mock_report(key))
    v = verify_sev_snp_chain(
        report, make_vcek(product="Turin-B0", hwid=bytes(8)), ask, ark
    )
    assert v.product == "Turin" and v.hwid_match
    assert decode_tcb(0x0700000004030201, "Turin") == {
        "fmc": 1, "bl": 2, "tee": 3, "snp": 4, "ucode": 7,
    }


def test_vcek_without_hwid_fails(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    v = verify_sev_snp_chain(
        parse_sev_snp_report(_mock_report(key)), make_vcek(hwid=None), ask, ark
    )
    assert not v.hwid_match and not v.ok


def test_attested_tier_needs_chain_and_pin(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    record = {"version": 1, "id": "rec-1"}
    report = _mock_report(key, bind_record_to_report_data(record))
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    chain = (make_vcek(), ask, ark)

    no_chain = verify_enforcement(record, report, pem,
                                  expected_measurement=MEASUREMENT.hex())
    assert no_chain.tier == "measurement_pinned"
    assert no_chain.vcek_chain_basis == "caller_supplied_unverified"

    unpinned = verify_enforcement(record, report, pem, amd_chain=chain)
    assert unpinned.tier == "bound" and unpinned.vcek_chain_basis == "kds_verified"

    attested = verify_enforcement(record, report, pem, amd_chain=chain,
                                  expected_measurement=MEASUREMENT.hex(), strict=True)
    assert attested.tier == "attested" and attested.ok

    bad = verify_enforcement(record, report, pem, amd_chain=(make_vcek(ucode=3), ask, ark))
    assert bad.vcek_chain_basis == "chain_failed" and not bad.ok


def test_set_counts_attested_entries(mock_chain, pin_mock_root):
    key, ark, ask, make_vcek = mock_chain
    record = {"version": 1, "id": "rec-2"}
    report = _mock_report(key, bind_record_to_report_data(record))
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    out = check_enforcement_set(
        [("a", record, report, pem, (make_vcek(), ask, ark)), ("b", record, report, pem)],
        expected_measurement=MEASUREMENT.hex(),
    )
    assert out.ok
    assert out.tier_counts["attested"] == 1
    assert out.tier_counts["measurement_pinned"] == 1


# ---- certificate table and PEM chain file ------------------------------------


def test_certificate_table_round_trip():
    guids = {name: guid for guid, name in SNP_CERT_GUIDS.items()}
    payloads = {"vcek": b"V" * 10, "ask": b"S" * 7, "ark": b"R" * 5}
    header_len = 24 * (len(payloads) + 1)
    table = bytearray()
    body = bytearray()
    for name, data in payloads.items():
        table += guids[name].bytes_le + struct.pack("<II", header_len + len(body), len(data))
        body += data
    table += bytes(24)
    assert parse_certificate_table(bytes(table + body)) == payloads


def test_certificate_table_skips_unknown_guids_and_rejects_overruns():
    unknown = uuid.uuid4().bytes_le + struct.pack("<II", 48, 1)
    assert parse_certificate_table(unknown + bytes(24) + b"x") == {}
    with pytest.raises(TEEAttestationError):
        parse_certificate_table(
            list(SNP_CERT_GUIDS)[0].bytes_le + struct.pack("<II", 48, 99) + bytes(24)
        )


def test_split_pem_chain():
    vcek_pem = chain_mod._load_cert(_fix("vcek.der")).public_bytes(
        serialization.Encoding.PEM
    )
    joined = b"# chain\n" + vcek_pem + _fix("ask.pem") + _fix("ark.pem")
    a, b, c = split_pem_chain(joined)
    assert certificate_product(a) == "Milan"
    assert chain_mod._load_cert(c).subject == chain_mod._load_cert(_fix("ark.pem")).subject
    with pytest.raises(TEEAttestationError):
        split_pem_chain(_fix("ask.pem"))


# ---- live emission -----------------------------------------------------------


def test_ioctl_number_matches_linux_uapi():
    # _IOWR('S', 0x0, struct snp_guest_request_ioctl), a 32-byte struct.
    from vaara.attestation.tee import _SNP_GET_REPORT

    assert _SNP_GET_REPORT == 0xC0205300


def test_host_attester_without_a_guest_raises(tmp_path):
    attester = SEVSNPHostAttester(
        str(tmp_path / "sev-guest"), tsm_root=str(tmp_path / "tsm")
    )
    with pytest.raises(TEEAttestationError, match="not an SEV-SNP guest"):
        attester.emit(bytes(64))
    with pytest.raises(TEEAttestationError, match="exactly 64"):
        attester.emit(b"short")


def _guest_device(tmp_path) -> str:
    device = tmp_path / "sev-guest"
    device.touch()
    return str(device)


def _fake_configfs(monkeypatch, report: bytes, aux: bytes, provider="sev_guest",
                   bump_generation=False):
    """Make mkdir under the fake tsm root behave like configfs-tsm."""
    real_mkdir = Path.mkdir
    written = {}

    def mkdir(self, *a, **k):
        real_mkdir(self, *a, **k)
        (self / "provider").write_text(provider + "\n")
        (self / "privlevel_floor").write_text("0\n")
        (self / "privlevel").write_text("")
        (self / "generation").write_text("1\n")
        (self / "outblob").write_bytes(report)
        (self / "auxblob").write_bytes(aux)
        written["dir"] = self

    real_read_bytes = Path.read_bytes

    def read_bytes(self):
        data = real_read_bytes(self)
        if bump_generation and self.name == "outblob":
            (self.parent / "generation").write_text("2\n")
        return data

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return written


def test_configfs_emission(tmp_path, monkeypatch):
    key = ec.generate_private_key(ec.SECP384R1())
    report = _mock_report(key, b"\x07" * 64)
    tsm = tmp_path / "tsm"
    tsm.mkdir()
    written = _fake_configfs(monkeypatch, report, b"aux")
    out, aux = SEVSNPHostAttester(
        _guest_device(tmp_path), tsm_root=str(tsm)
    ).emit_with_certificates(b"\x07" * 64)
    assert out == report and aux == b"aux"
    assert (written["dir"] / "inblob").read_bytes() == b"\x07" * 64
    assert (written["dir"] / "privlevel").read_text(encoding="utf-8") == "0"


def test_configfs_rejects_another_provider(tmp_path, monkeypatch):
    tsm = tmp_path / "tsm"
    tsm.mkdir()
    _fake_configfs(monkeypatch, bytes(1184), b"", provider="tdx_guest")
    with pytest.raises(TEEAttestationError, match="not sev_guest"):
        SEVSNPHostAttester(_guest_device(tmp_path), tsm_root=str(tsm)).emit(bytes(64))


def test_configfs_detects_a_concurrent_write(tmp_path, monkeypatch):
    tsm = tmp_path / "tsm"
    tsm.mkdir()
    _fake_configfs(monkeypatch, bytes(1184), b"", bump_generation=True)
    with pytest.raises(TEEAttestationError, match="changed while"):
        SEVSNPHostAttester(_guest_device(tmp_path), tsm_root=str(tsm)).emit(bytes(64))


def test_configfs_without_the_sev_guest_device_is_not_a_guest(tmp_path):
    # CI runners expose configfs-tsm with no SEV-SNP guest behind it.
    tsm = tmp_path / "tsm"
    tsm.mkdir()
    attester = SEVSNPHostAttester(str(tmp_path / "sev-guest"), tsm_root=str(tsm))
    with pytest.raises(TEEAttestationError, match="not an SEV-SNP guest"):
        attester.emit(bytes(64))


# ---- report versions ---------------------------------------------------------


@pytest.mark.parametrize("version", [3, 5])
def test_newer_report_versions_bind(version):
    key = ec.generate_private_key(ec.SECP384R1())
    record = {"version": 1, "id": f"v{version}"}
    report = MockSEVSNPAttester(key, measurement=MEASUREMENT, version=version).emit(
        bind_record_to_report_data(record)
    )
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    v = verify_enforcement(record, report, pem)
    assert v.report_version == version and v.tier == "bound" and v.ok


def test_cli_verify_enforcement_with_amd_chain(tmp_path, capsys):
    from vaara.cli import main

    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"version": 1}))
    vcek_pem = tmp_path / "vcek.pem"
    vcek_pem.write_bytes(_vcek_pem(_fix("vcek.der")))
    rc = main([
        "verify-enforcement", str(rec), "--report", str(FIX / "report.bin"),
        "--vcek", str(vcek_pem), "--amd-chain", str(FIX / "vcek.der"),
        str(FIX / "ask.pem"), str(FIX / "ark.pem"), "--json",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1  # the report binds to another record
    assert out["vcek_chain_basis"] == "kds_verified"
    assert out["amd_chain"]["product"] == "Milan"


def _combined_chain(tmp_path) -> Path:
    vcek = chain_mod._load_cert(_fix("vcek.der")).public_bytes(serialization.Encoding.PEM)
    path = tmp_path / "chain.pem"
    path.write_bytes(vcek + _fix("ask.pem") + _fix("ark.pem"))
    return path


def test_cli_tee_verify_with_combined_chain(tmp_path, capsys):
    from vaara.cli import main

    vcek_pem = tmp_path / "vcek.pem"
    vcek_pem.write_bytes(_vcek_pem(_fix("vcek.der")))
    rc = main(["tee", "verify", str(FIX / "report.bin"), "--vcek", str(vcek_pem),
               "--amd-chain", str(_combined_chain(tmp_path))])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["amd_chain"]["ok"]


def test_cli_tee_verify_fails_on_a_foreign_root(tmp_path, capsys):
    from vaara.cli import main

    vcek_pem = tmp_path / "vcek.pem"
    vcek_pem.write_bytes(_vcek_pem(_fix("vcek.der")))
    rc = main(["tee", "verify", str(FIX / "report.bin"), "--vcek", str(vcek_pem),
               "--amd-chain", str(FIX / "vcek.der"), str(FIX / "genoa_ask.pem"),
               str(FIX / "genoa_ark.pem")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and not out["amd_chain"]["ark_pinned"]


def test_cli_amd_chain_needs_one_or_three_files(tmp_path, capsys):
    from vaara.cli import main

    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"version": 1}))
    vcek_pem = tmp_path / "vcek.pem"
    vcek_pem.write_bytes(_vcek_pem(_fix("vcek.der")))
    rc = main(["verify-enforcement", str(rec), "--report", str(FIX / "report.bin"),
               "--vcek", str(vcek_pem), "--amd-chain", str(FIX / "ask.pem"),
               str(FIX / "ark.pem")])
    assert rc == 1
    assert "one combined PEM file or three" in capsys.readouterr().err


def test_cli_tee_emit_outside_a_guest(tmp_path, capsys, monkeypatch):
    from vaara.cli import main
    from vaara.attestation import tee

    def not_a_guest(self, report_data):
        raise TEEAttestationError("not present. This host is not an SEV-SNP guest.")

    monkeypatch.setattr(tee.SEVSNPHostAttester, "emit_with_certificates", not_a_guest)
    rc = main(["tee", "emit", "--report-data", "00" * 64,
               "--out", str(tmp_path / "r.bin")])
    assert rc == 1 and "not an SEV-SNP guest" in capsys.readouterr().err
    assert not (tmp_path / "r.bin").exists()


def test_cli_tee_emit_writes_report_and_chain(tmp_path, capsys, monkeypatch):
    from vaara.cli import main
    from vaara.attestation import tee

    guids = {name: guid for guid, name in SNP_CERT_GUIDS.items()}
    parts = {"vcek": _fix("vcek.der"), "ask": _fix("ask.pem"), "ark": _fix("ark.pem")}
    header = 24 * 4
    table, body = bytearray(), bytearray()
    for name, data in parts.items():
        table += guids[name].bytes_le + struct.pack("<II", header + len(body), len(data))
        body += data
    table += bytes(24)

    monkeypatch.setattr(
        tee.SEVSNPHostAttester, "emit_with_certificates",
        lambda self, rd: (_fix("report.bin"), bytes(table + body)),
    )
    rc = main(["tee", "emit", "--report-data", "00" * 64,
               "--out", str(tmp_path / "r.bin"), "--certs-out", str(tmp_path / "c.pem")])
    assert rc == 0
    a, b, c = split_pem_chain((tmp_path / "c.pem").read_bytes())
    v = verify_sev_snp_chain(parse_sev_snp_report((tmp_path / "r.bin").read_bytes()), a, b, c)
    assert v.ok, v.reason


def test_set_discovery_reads_amd_chain_file(tmp_path):
    from vaara.cli import _discover_enforcement_triples

    (tmp_path / "x.record.json").write_text(json.dumps({"version": 1}))
    (tmp_path / "x.report.bin").write_bytes(_fix("report.bin"))
    (tmp_path / "x.vcek.pem").write_bytes(_vcek_pem(_fix("vcek.der")))
    (tmp_path / "x.amd-chain.pem").write_bytes(_combined_chain(tmp_path).read_bytes())
    triples, missing = _discover_enforcement_triples(
        [tmp_path / "x.record.json"], tmp_path
    )
    assert not missing and len(triples[0]) == 5
