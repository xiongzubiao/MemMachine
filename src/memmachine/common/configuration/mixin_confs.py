"""Metrics and authentication configuration mixins."""

import asyncio
import os
import re
from datetime import timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, ClassVar, Self

import requests
import yaml
from authlib.integrations.requests_client import OAuth2Session
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)

from memmachine.common.errors import InvalidPasswordError
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)


class UnknownMetricsFactoryError(ValueError):
    """Raised when the metrics factory name is invalid."""


class WithMetricsFactory:
    """Runtime mixin that provides access to a metrics factory."""

    _factories: ClassVar[dict[str, MetricsFactory]] = {}

    # These are *protocol attributes* — provided by subclasses
    metrics_factory_id: str | None

    def get_metrics_factory(self) -> MetricsFactory:
        """Return the configured metrics factory instance."""
        factory_id = self.metrics_factory_id or "prometheus"

        if factory_id not in self._factories:
            match factory_id:
                case "prometheus":
                    self._factories[factory_id] = PrometheusMetricsFactory()
                case _:
                    raise UnknownMetricsFactoryError(
                        f"Unknown MetricsFactory name: {factory_id}"
                    )

        return self._factories[factory_id]


class MetricsFactoryIdMixin(WithMetricsFactory, BaseModel):
    """Pydantic mixin for configs that include a metrics factory ID."""

    metrics_factory_id: str | None = Field(
        default=None,
        description="Metrics factory ID for monitoring and metrics collection.",
    )

    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined labels for metrics.",
    )


class WithValueFromEnv:
    """Mixin that adds support for resolving environment variable references."""

    # Matches $ENV or ${ENV}
    _ENV_RE: ClassVar[re.Pattern] = re.compile(r"\$(\w+)|\$\{(\w+)}")

    @classmethod
    def _resolve_env(cls, value: SecretStr | str) -> str:
        """Resolve environment variable references in the form $ENV or ${ENV}."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()

        if not isinstance(value, str):
            return value

        def _repl(match: re.Match) -> str:
            # One of the groups will be None
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))

        return cls._ENV_RE.sub(_repl, value)


class PasswordMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include a password.

    It reads the password from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    password: SecretStr = Field(
        ...,
        description="Password for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )

    @field_validator("password", mode="before")
    @classmethod
    def resolve_password(cls, v: str | SecretStr) -> SecretStr | str | None:
        """Resolve environment variable references in the password."""
        v = cls._resolve_env(v)
        if not isinstance(v, str):
            raise InvalidPasswordError("password must be a string or SecretStr")
        return SecretStr(v) if isinstance(v, str) else v


class AWSCredentialsMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include AWS credentials.

    It reads the credentials from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    aws_access_key_id: SecretStr | None = Field(
        default=None,
        description="AWS Access Key ID. It default to environment variable "
        "AWS_ACCESS_KEY_ID if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None,
        description="AWS Secret Access Key. It defaults to environment variable "
        "AWS_SECRET_ACCESS_KEY if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )
    aws_session_token: SecretStr | None = Field(
        default=None,
        description="AWS session token for authentication. It defaults to environment variable "
        "AWS_SESSION_TOKEN if not provided. Can reference an "
        "environment variable using $ENV_NAME syntax.",
    )

    @field_validator("aws_access_key_id", mode="before")
    @classmethod
    def resolve_aws_access_key_id(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Access Key ID."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("aws_secret_access_key", mode="before")
    @classmethod
    def resolve_aws_secret_access_key(
        cls, v: SecretStr | str
    ) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Secret Access Key."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("aws_session_token", mode="before")
    @classmethod
    def resolve_aws_session_token(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the AWS Session Token."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def resolve_aws_env_defaults(self) -> Self:
        """Fill in AWS credentials from environment variables if not provided."""
        if not self.aws_access_key_id:
            v = os.getenv("AWS_ACCESS_KEY_ID", None)
            if v:
                self.aws_access_key_id = SecretStr(v)

        if not self.aws_secret_access_key:
            v = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            if v:
                self.aws_secret_access_key = SecretStr(v)

        if not self.aws_session_token:
            v = os.getenv("AWS_SESSION_TOKEN", None)
            if v:
                self.aws_session_token = SecretStr(v)

        return self


class AuthMixin(BaseModel, WithValueFromEnv):
    """
    Mixin for configurations that include either an API key or OAuth credentials.

    It reads credentials from environment variables if user
    specifies a pattern like $ENV_NAME in the value.
    """

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )

    issuer_url: str = Field(
        default="",
        description="OAuth issuer URL for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )
    client_id: SecretStr = Field(
        default=SecretStr(""),
        description="OAuth client ID for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )
    client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="OAuth client secret for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )
    refresh_token: SecretStr = Field(
        default=SecretStr(""),
        description="OAuth refresh token for authentication.  Can reference an environment variable using $ENV_NAME syntax.",
    )

    _oauth_session: OAuth2Session | None = PrivateAttr(default=None)

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the API key."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("client_id", mode="before")
    @classmethod
    def resolve_client_id(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the OAuth client ID."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("client_secret", mode="before")
    @classmethod
    def resolve_client_secret(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the OAuth client secret."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @field_validator("issuer_url", mode="before")
    @classmethod
    def resolve_issuer_url(cls, v: str) -> str:
        """Resolve environment variable references in issuer URL."""
        resolved = cls._resolve_env(v)
        if isinstance(resolved, str):
            return resolved
        return v

    @field_validator("refresh_token", mode="before")
    @classmethod
    def resolve_refresh_token(cls, v: SecretStr | str) -> SecretStr | str | None:
        """Resolve environment variable references in the OAuth refresh token."""
        v = cls._resolve_env(v)
        return SecretStr(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> Self:
        # API key takes precedence
        if (
            not self.api_key.get_secret_value()
            and self.refresh_token.get_secret_value()
        ):
            if not self.client_id.get_secret_value():
                raise ValueError(
                    "OAuth client_id must be set when refresh_token is used"
                )
            if not self.issuer_url:
                raise ValueError(
                    "OAuth issuer_url must be set when refresh_token is used"
                )

        return self

    def resolve_auth(self) -> str | Callable[[], Awaitable[str]] | None:
        api_key = self.api_key.get_secret_value()
        if api_key:
            return api_key
        refresh_token = self.refresh_token.get_secret_value()
        if refresh_token:
            return self.fetch_access_token
        return None

    async def _resolve_openid_config(self) -> dict[str, Any]:
        metadata_url = f"{self.issuer_url.rstrip('/')}/.well-known/openid-configuration"

        try:
            response = await asyncio.to_thread(
                requests.get,
                metadata_url,
                timeout=10,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"OAuth metadata request failed for {metadata_url}"
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to load OAuth metadata from {metadata_url} (status {response.status_code})"
            )

        try:
            metadata = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OAuth metadata from {metadata_url} is not valid JSON"
            ) from exc

        if not isinstance(metadata, dict):
            raise TypeError(f"OAuth metadata from {metadata_url} is not a JSON object")

        return metadata

    async def _get_oauth_session(self) -> OAuth2Session:
        if self._oauth_session is not None:
            return self._oauth_session

        openid_config = await self._resolve_openid_config()
        token_endpoint = openid_config.get("token_endpoint")
        if not token_endpoint:
            raise RuntimeError("Token endpoint not found in OpenID configuration")

        client_id = self.client_id.get_secret_value()
        client_secret = self.client_secret.get_secret_value()
        refresh_token = self.refresh_token.get_secret_value()

        token_endpoint_auth_method = "none"
        if client_secret:
            token_endpoint_auth_method = "client_secret_post"

        session = OAuth2Session(
            client_id=client_id,
            client_secret=client_secret or None,
            token_endpoint_auth_method=token_endpoint_auth_method,
            token_endpoint=token_endpoint,
            token={"refresh_token": refresh_token, "token_type": "Bearer"},
            default_timeout=10,
        )

        self._oauth_session = session
        return session

    async def fetch_access_token(self) -> str:
        """Exchange or refresh OAuth tokens to obtain a valid access token."""
        session = await self._get_oauth_session()
        token_endpoint = session.metadata.get("token_endpoint")

        if not session.token.get("access_token"):
            await asyncio.to_thread(
                session.refresh_token,
                token_endpoint,
                refresh_token=self.refresh_token.get_secret_value(),
            )
        else:
            # Subsequent calls: auto-refresh if token expired
            await asyncio.to_thread(session.ensure_active_token)

        access_token = session.token.get("access_token")
        if not access_token:
            raise RuntimeError("OAuth session has no access token")

        return access_token


class YamlSerializableMixin(BaseModel):
    """Mixin that adds YAML-safe serialization for Pydantic models."""

    def to_yaml_dict(self) -> dict:
        raw = self.model_dump()

        def unwrap(obj: YamlObjType) -> YamlObjType:
            """Recursively unwrap Pydantic models, SecretStr, Enums, and drop empty values."""
            if isinstance(obj, YamlSerializableMixin):
                obj = obj.to_yaml_dict()

            # Unwrap SecretStr
            if isinstance(obj, SecretStr):
                obj = obj.get_secret_value()

            # Unwrap enums like SimilarityMetric
            if isinstance(obj, Enum):
                obj = obj.value

            if isinstance(obj, timedelta):
                obj = obj.total_seconds()

            # Dict — recurse & drop empty
            if isinstance(obj, dict):
                cleaned = {k: unwrap(v) for k, v in obj.items()}
                # drop keys whose values are None/empty
                cleaned = {
                    k: v for k, v in cleaned.items() if v not in (None, "", [], {})
                }
                return cleaned

            # List — recurse & drop empty
            if isinstance(obj, list):
                cleaned = [unwrap(v) for v in obj]
                cleaned = [v for v in cleaned if v not in (None, "", [], {})]
                return cleaned

            # Base condition
            return obj

        ret = unwrap(raw)
        if not isinstance(ret, dict):
            raise TypeError(
                "to_yaml_dict can only be called on models that serialize to dicts"
            )
        return ret

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_yaml_dict(), sort_keys=False)


type YamlObjType = (
    YamlSerializableMixin
    | SecretStr
    | Enum
    | dict[str, "YamlObjType"]
    | list["YamlObjType"]
    | str
    | int
    | float
    | bool
    | None
)
