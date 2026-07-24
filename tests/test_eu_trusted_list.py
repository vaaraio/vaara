# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parse the EU LOTL and national trusted lists (ETSI TS 119 612) down to the
qualified timestamping services a user can pick as an anchor provider.
"""
from vaara.audit.eu_trusted_list import (
    LOTL_URL,
    QualifiedTSA,
    TSLPointer,
    parse_lotl,
    parse_trusted_list,
    providers_for_country,
)

# Minimal LOTL: one pointer to an XML national list (AT), one to a PDF (human
# readable) that must be ignored.
LOTL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TrustServiceStatusList xmlns="http://uri.etsi.org/02231/v2#">
  <SchemeInformation>
    <PointersToOtherTSL>
      <OtherTSLPointer>
        <TSLLocation>https://tl.example.at/tl.xml</TSLLocation>
        <AdditionalInformation>
          <OtherInformation><SchemeTerritory>AT</SchemeTerritory></OtherInformation>
          <OtherInformation><MimeType xmlns="http://uri.etsi.org/02231/v2/additionaltypes#">application/vnd.etsi.tsl+xml</MimeType></OtherInformation>
        </AdditionalInformation>
      </OtherTSLPointer>
      <OtherTSLPointer>
        <TSLLocation>https://tl.example.at/tl.pdf</TSLLocation>
        <AdditionalInformation>
          <OtherInformation><SchemeTerritory>AT</SchemeTerritory></OtherInformation>
          <OtherInformation><MimeType xmlns="http://uri.etsi.org/02231/v2/additionaltypes#">application/pdf</MimeType></OtherInformation>
        </AdditionalInformation>
      </OtherTSLPointer>
    </PointersToOtherTSL>
  </SchemeInformation>
</TrustServiceStatusList>
"""

# Minimal national list: one provider, one qualified timestamp service (granted,
# with an endpoint) and one non-timestamp service that must be excluded.
AT_TL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TrustServiceStatusList xmlns="http://uri.etsi.org/02231/v2#">
  <TrustServiceProviderList>
    <TrustServiceProvider>
      <TSPInformation>
        <TSPName><Name xml:lang="en">ACME QTSP</Name></TSPName>
      </TSPInformation>
      <TSPServices>
        <TSPService><ServiceInformation>
          <ServiceTypeIdentifier>http://uri.etsi.org/TrstSvc/Svctype/TSA/QTST</ServiceTypeIdentifier>
          <ServiceName><Name xml:lang="en">ACME Qualified Timestamping</Name></ServiceName>
          <ServiceStatus>http://uri.etsi.org/TrstSvc/TrustedList/Svcstatus/granted</ServiceStatus>
          <ServiceSupplyPoints>
            <ServiceSupplyPoint>https://tsa.example.at/tsa</ServiceSupplyPoint>
          </ServiceSupplyPoints>
        </ServiceInformation></TSPService>
        <TSPService><ServiceInformation>
          <ServiceTypeIdentifier>http://uri.etsi.org/TrstSvc/Svctype/CA/QC</ServiceTypeIdentifier>
          <ServiceName><Name xml:lang="en">ACME Qualified Certificates</Name></ServiceName>
          <ServiceStatus>http://uri.etsi.org/TrstSvc/TrustedList/Svcstatus/granted</ServiceStatus>
        </ServiceInformation></TSPService>
      </TSPServices>
    </TrustServiceProvider>
  </TrustServiceProviderList>
</TrustServiceStatusList>
"""


def test_parse_lotl_returns_only_xml_national_list_pointers():
    pointers = parse_lotl(LOTL_XML)
    assert pointers == [TSLPointer(territory="AT", location="https://tl.example.at/tl.xml")]


def test_parse_trusted_list_returns_only_qualified_timestamp_services():
    tsas = parse_trusted_list(AT_TL_XML, territory="AT")
    assert tsas == [
        QualifiedTSA(
            territory="AT",
            provider="ACME QTSP",
            service_name="ACME Qualified Timestamping",
            endpoint="https://tsa.example.at/tsa",
        )
    ]


def test_parse_trusted_list_skips_non_granted_services():
    revoked = AT_TL_XML.replace(b"Svcstatus/granted", b"Svcstatus/withdrawn")
    assert parse_trusted_list(revoked, territory="AT") == []


def _fake_fetch(url: str) -> bytes:
    return {
        LOTL_URL: LOTL_XML,
        "https://tl.example.at/tl.xml": AT_TL_XML,
    }[url]


def test_providers_for_country_walks_lotl_to_national_list():
    tsas = providers_for_country("AT", fetch=_fake_fetch)
    assert tsas == [
        QualifiedTSA(
            territory="AT",
            provider="ACME QTSP",
            service_name="ACME Qualified Timestamping",
            endpoint="https://tsa.example.at/tsa",
        )
    ]


def test_providers_for_country_unknown_country_is_empty():
    assert providers_for_country("ZZ", fetch=_fake_fetch) == []
