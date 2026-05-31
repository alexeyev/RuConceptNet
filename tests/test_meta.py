"""Tests for ruconceptnet.meta and top-level package attributes."""

import ruconceptnet
import ruconceptnet.meta as meta


class TestMeta:
    def test_version_is_string(self):
        assert isinstance(meta.version, str)

    def test_version_non_empty(self):
        assert len(meta.version) > 0

    def test_authors_is_list(self):
        assert isinstance(meta.authors, list)

    def test_authors_non_empty(self):
        assert len(meta.authors) > 0

    def test_license_is_string(self):
        assert isinstance(meta.license, str)

    def test_copyright_is_string(self):
        assert isinstance(meta.copyright, str)

    def test_copyright_contains_author(self):
        assert meta.authors[0] in meta.copyright

    def test_emails_is_list(self):
        assert isinstance(meta.emails, list)

    def test_emails_non_empty(self):
        assert len(meta.emails) > 0


class TestPackageExports:
    def test_version_exported(self):
        assert hasattr(ruconceptnet, "__version__")
        assert ruconceptnet.__version__ == meta.version

    def test_author_exported(self):
        assert hasattr(ruconceptnet, "__author__")
        assert ruconceptnet.__author__ == meta.authors[0]

    def test_license_exported(self):
        assert hasattr(ruconceptnet, "__license__")
        assert ruconceptnet.__license__ == meta.license

    def test_copyright_exported(self):
        assert hasattr(ruconceptnet, "__copyright__")

    def test_conceptnet_class_exported(self):
        assert hasattr(ruconceptnet, "ConceptNet")
