# Render the Vaara receipt Internet-Draft.
# Requires: pip install xml2rfc  (3.34.0 or newer)

DRAFT = draft-sirkkavaara-vaara-receipt-05
XML   = $(DRAFT).xml

.PHONY: all text html clean

all: text html

text: $(XML)
	xml2rfc --text $(XML)

html: $(XML)
	xml2rfc --html $(XML)

clean:
	rm -f $(DRAFT).txt $(DRAFT).html
