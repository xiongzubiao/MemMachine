import pytest
import yaml
from pydantic import SecretStr

from memmachine.common.configuration.database_conf import (
    DatabasesConf,
    SqlAlchemyConf,
    SupportedDB,
)


def test_parse_supported_db_enums():
    assert SupportedDB.from_provider("postgres") == SupportedDB.POSTGRES
    assert SupportedDB.from_provider("sqlite") == SupportedDB.SQLITE

    pg_db = SupportedDB.POSTGRES
    assert pg_db.conf_cls == SqlAlchemyConf
    assert pg_db.dialect == "postgresql"
    assert pg_db.driver == "asyncpg"

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
def db_conf_dict() -> dict:
    return {
        "databases": {
            "main_postgres": {
                "provider": "postgres",
                "config": {
                    "host": "db.example.com",
                    "port": 5432,
                    "user": "admin",
                    "password": "pwd",
                    "db_name": "test_db",
                },
            },
            "local_sqlite": {
                "provider": "sqlite",
                "config": {
                    "path": "local.db",
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

    # Postgres check
    pg_conf = storage_conf.relational_db_confs["main_postgres"]
    assert isinstance(pg_conf, SqlAlchemyConf)
    assert pg_conf.dialect == "postgresql"
    assert pg_conf.driver == "asyncpg"
    assert pg_conf.host == "db.example.com"
    assert pg_conf.user == "admin"
    assert pg_conf.password == SecretStr("pwd")
    assert pg_conf.db_name == "test_db"
    assert pg_conf.port == 5432
    assert pg_conf.path is None
    assert pg_conf.uri == "postgresql+asyncpg://admin:pwd@db.example.com:5432/test_db"

    # Sqlite check
    sqlite_conf = storage_conf.relational_db_confs["local_sqlite"]
    assert sqlite_conf.dialect == "sqlite"
    assert sqlite_conf.driver == "aiosqlite"
    assert sqlite_conf.path == "local.db"
    assert isinstance(sqlite_conf, SqlAlchemyConf)
    assert sqlite_conf.uri == "sqlite+aiosqlite:///local.db"


def test_read_db_password_from_env(monkeypatch, db_conf_dict):
    monkeypatch.setenv("MY_DB_PASSWORD", "env-db-password")
    db_conf_dict["databases"]["main_postgres"]["config"]["password"] = (
        "${MY_DB_PASSWORD}"
    )
    storage_conf = DatabasesConf.parse(db_conf_dict)

    pg_conf = storage_conf.relational_db_confs["main_postgres"]
    assert pg_conf.password == SecretStr("env-db-password")


def test_parse_unknown_provider_raises():
    input_dict = {
        "databases": {"bad_storage": {"provider": "unknown_db", "host": "localhost"}},
    }
    message = "Supported providers are: postgres, sqlite"
    with pytest.raises(ValueError, match=message):
        DatabasesConf.parse(input_dict)


def test_parse_empty_storage_returns_empty_conf():
    input_dict = {"databases": {}}
    storage_conf = DatabasesConf.parse(input_dict)
    assert storage_conf.relational_db_confs == {}


def test_serialize_deserialize_database_conf(db_conf_dict):
    conf = DatabasesConf.parse(db_conf_dict)
    yaml_str = conf.to_yaml()
    conf_cp = DatabasesConf.parse(yaml.safe_load(yaml_str))
    assert conf == conf_cp
