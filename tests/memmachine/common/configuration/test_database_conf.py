import pytest
import yaml

from memmachine.common.configuration.database_conf import (
    ChromaConf,
    DatabasesConf,
    SqlAlchemyConf,
    SupportedDB,
)


def test_parse_supported_db_enums():
    assert SupportedDB.from_provider("sqlite") == SupportedDB.SQLITE

    sqlite_db = SupportedDB.SQLITE
    assert sqlite_db.conf_cls == SqlAlchemyConf
    assert sqlite_db.dialect == "sqlite"
    assert sqlite_db.driver == "aiosqlite"


def test_sqlite_without_path_raises():
    message = "non-empty 'path'"
    with pytest.raises(ValueError, match=message):
        SupportedDB.SQLITE.build_config({"uri": "sqlite.db"})


def test_sqlite_with_path_succeeds():
    config = SupportedDB.SQLITE.build_config({"path": "sqlite.db"})
    assert isinstance(config, SqlAlchemyConf)
    assert config.path == "sqlite.db"
    assert config.uri == "sqlite+aiosqlite:///sqlite.db"


def test_invalid_provider_raises():
    message = "Supported providers are"
    with pytest.raises(ValueError, match=message):
        SupportedDB.from_provider("invalid_db")


@pytest.fixture
def db_conf_dict() -> dict[str, object]:
    return {
        "databases": {
            "local_sqlite": {
                "provider": "sqlite",
                "config": {
                    "path": "local.db",
                },
            },
            "semantic_chroma": {
                "provider": "chroma",
                "config": {
                    "path": "/tmp/chroma",
                    "collection_prefix": "memmachine",
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ["MY_DB_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)


def test_parse_valid_storage_dict(db_conf_dict):
    storage_conf = DatabasesConf.parse(db_conf_dict)

    # Sqlite check
    sqlite_conf = storage_conf.relational_db_confs["local_sqlite"]
    assert sqlite_conf.dialect == "sqlite"
    assert sqlite_conf.driver == "aiosqlite"
    assert sqlite_conf.path == "local.db"
    assert isinstance(sqlite_conf, SqlAlchemyConf)
    assert sqlite_conf.uri == "sqlite+aiosqlite:///local.db"

    chroma_conf = storage_conf.vector_db_confs["semantic_chroma"]
    assert isinstance(chroma_conf, ChromaConf)
    assert chroma_conf.path == "/tmp/chroma"
    assert chroma_conf.collection_prefix == "memmachine"


def test_parse_unknown_provider_raises():
    input_dict = {
        "databases": {"bad_storage": {"provider": "unknown_db", "host": "localhost"}},
    }
    message = "Supported providers are: sqlite, chroma"
    with pytest.raises(ValueError, match=message):
        DatabasesConf.parse(input_dict)


def test_parse_empty_storage_returns_empty_conf():
    input_dict = {"databases": {}}
    storage_conf = DatabasesConf.parse(input_dict)
    assert storage_conf.relational_db_confs == {}
    assert storage_conf.vector_db_confs == {}


def test_serialize_deserialize_database_conf(db_conf_dict):
    conf = DatabasesConf.parse(db_conf_dict)
    yaml_str = conf.to_yaml()
    conf_cp = DatabasesConf.parse(yaml.safe_load(yaml_str))
    assert conf == conf_cp
