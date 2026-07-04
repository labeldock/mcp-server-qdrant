from mcp_server_qdrant.common.permissions import (
    parse_collection_directives,
    union_permissions,
)
from mcp_server_qdrant.settings import QdrantSettings


class TestParseCollectionDirectives:
    def test_empty_is_no_whitelist(self):
        assert parse_collection_directives(None) == {}
        assert parse_collection_directives("") == {}
        assert parse_collection_directives("   ") == {}

    def test_bare_name_defaults_to_full(self):
        access = parse_collection_directives("mycol")
        assert set(access) == {"mycol"}
        assert access["mycol"].perm_string() == "rwd"

    def test_directives_parse_permissions(self):
        access = parse_collection_directives("travel:ro place:rw pin:rwd")
        assert list(access) == ["travel", "place", "pin"]
        assert access["travel"].perm_string() == "r--"
        assert access["place"].perm_string() == "rw-"
        assert access["pin"].perm_string() == "rwd"

    def test_ro_is_read_only(self):
        access = parse_collection_directives("c:ro")
        assert access["c"].can("r")
        assert not access["c"].can("w")
        assert not access["c"].can("d")

    def test_unknown_letters_ignored(self):
        # 'x' is ignored; 'r' still counts.
        access = parse_collection_directives("c:rx")
        assert access["c"].perm_string() == "r--"

    def test_empty_perms_after_colon_defaults_to_full(self):
        access = parse_collection_directives("c:")
        assert access["c"].perm_string() == "rwd"

    def test_read_only_override_strips_write_and_delete(self):
        access = parse_collection_directives("a:rwd b:rw", read_only=True)
        assert access["a"].perm_string() == "r--"
        assert access["b"].perm_string() == "r--"

    def test_perm_words(self):
        access = parse_collection_directives("c:rwd")
        assert access["c"].perm_words() == ["read", "write", "delete"]

    def test_union_permissions(self):
        access = parse_collection_directives("a:ro b:rw c:rwd")
        assert union_permissions(access) == frozenset({"r", "w", "d"})

    def test_union_permissions_read_only_collections(self):
        access = parse_collection_directives("a:ro b:ro")
        assert union_permissions(access) == frozenset({"r"})


class TestQdrantSettingsAccess:
    def test_single_collection_is_default(self, monkeypatch):
        monkeypatch.setenv("COLLECTION_NAME", "pin:rwd")
        settings = QdrantSettings()
        assert settings.default_collection_name() == "pin"

    def test_multiple_collections_have_no_default(self, monkeypatch):
        monkeypatch.setenv("COLLECTION_NAME", "travel:ro place:rw")
        settings = QdrantSettings()
        assert settings.default_collection_name() is None
        assert set(settings.collection_access()) == {"travel", "place"}

    def test_read_only_env_applies_to_access(self, monkeypatch):
        monkeypatch.setenv("COLLECTION_NAME", "a:rwd")
        monkeypatch.setenv("QDRANT_READ_ONLY", "true")
        settings = QdrantSettings()
        assert settings.collection_access()["a"].perm_string() == "r--"

    def test_mcp_password_defaults_none(self, monkeypatch):
        monkeypatch.delenv("MCP_PASSWORD", raising=False)
        settings = QdrantSettings()
        assert settings.mcp_password is None

    def test_mcp_password_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_PASSWORD", "s3cr3t")
        settings = QdrantSettings()
        assert settings.mcp_password == "s3cr3t"
