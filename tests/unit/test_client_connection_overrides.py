from copy import deepcopy
from unittest.mock import patch

import pytest

from llm_exec_core.client import LLMClient

PRIMARY_KEY = "SYNTHETIC_PRIMARY_API_KEY"
FIRST_ALIAS = "SYNTHETIC_FIRST_API_KEY"
SECOND_ALIAS = "SYNTHETIC_SECOND_API_KEY"
ENDPOINT_VARIABLE = "SYNTHETIC_API_BASE_URL"
STATIC_URL = "https://static.example.invalid/v1/chat/completions"
OMITTED = object()


class TrackingEnvironment(dict):
    def __init__(self, values):
        super().__init__(values)
        self.lookups = []

    def get(self, key, default=None):
        self.lookups.append(key)
        return super().get(key, default)


def _catalog(
    *,
    aliases=OMITTED,
    endpoint_variable=OMITTED,
    static_url=STATIC_URL,
    provider_name="synthetic-provider",
):
    provider = {
        "api_key_env_var": PRIMARY_KEY,
        "api_base_url": static_url,
        "temperature": 0.1,
        "max_tokens": 128,
        "context_window": 4096,
        "pricing_currency": "$",
        "models": {
            "synthetic-model": {
                "id": "synthetic-model-id",
                "pricing": {"input": 1.0, "output": 2.0},
            }
        },
    }
    if aliases is not OMITTED:
        provider["api_key_env_aliases"] = aliases
    if endpoint_variable is not OMITTED:
        provider["api_base_url_env_var"] = endpoint_variable
    return {provider_name: provider}


def _client_with_environment(environment, catalog=None):
    with patch("llm_exec_core.client.os.environ", environment):
        return LLMClient(
            "synthetic-model", config_source=catalog or _catalog()
        )


@pytest.mark.parametrize("primary_value", [None, "", "\t\u2003"])
def test_no_alias_missing_or_blank_primary_preserves_legacy_error(
    primary_value,
):
    environment = {}
    if primary_value is not None:
        environment[PRIMARY_KEY] = primary_value

    with pytest.raises(ValueError) as error:
        _client_with_environment(environment)

    assert str(error.value) == (
        f"API key environment variable '{PRIMARY_KEY}' is not set."
    )


@pytest.mark.parametrize(
    ("environment", "expected_key"),
    [
        (
            {
                PRIMARY_KEY: "primary-key",
                FIRST_ALIAS: "first-alias",
                SECOND_ALIAS: "second-alias",
            },
            "primary-key",
        ),
        (
            {FIRST_ALIAS: "first-alias", SECOND_ALIAS: "second-alias"},
            "first-alias",
        ),
        (
            {
                PRIMARY_KEY: "",
                FIRST_ALIAS: "\u2003\t",
                SECOND_ALIAS: "  second-alias  ",
            },
            "  second-alias  ",
        ),
    ],
)
def test_api_key_resolution_uses_declared_precedence_and_original_value(
    environment, expected_key
):
    client = _client_with_environment(
        environment,
        _catalog(aliases=[FIRST_ALIAS, SECOND_ALIAS]),
    )

    assert client.api_key == expected_key
    assert client.headers["Authorization"] == f"Bearer {expected_key}"
    assert client.provider_name == "synthetic-provider"


def test_api_key_lookup_stops_after_first_populated_alias():
    environment = TrackingEnvironment(
        {
            PRIMARY_KEY: "",
            FIRST_ALIAS: "winner",
            SECOND_ALIAS: "later-value",
        }
    )

    client = _client_with_environment(
        environment,
        _catalog(
            aliases=[FIRST_ALIAS, PRIMARY_KEY, FIRST_ALIAS, SECOND_ALIAS]
        ),
    )

    assert client.api_key == "winner"
    assert environment.lookups == [PRIMARY_KEY, FIRST_ALIAS]


def test_all_missing_or_blank_alias_error_names_declarations_in_order():
    environment = {
        PRIMARY_KEY: "",
        FIRST_ALIAS: "\u2003",
        SECOND_ALIAS: "\t",
    }

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            environment,
            _catalog(aliases=[FIRST_ALIAS, SECOND_ALIAS]),
        )

    assert str(error.value) == (
        "API key environment variables "
        f"'{PRIMARY_KEY}', '{FIRST_ALIAS}', '{SECOND_ALIAS}' "
        "are not set or are blank."
    )


@pytest.mark.parametrize("control_code", [*range(32), 127])
def test_selected_api_key_rejects_every_c0_and_del_without_value_leak(
    control_code,
):
    secret_value = f"secret-prefix{chr(control_code)}secret-suffix"

    with pytest.raises(ValueError) as error:
        _client_with_environment({PRIMARY_KEY: secret_value})

    message = str(error.value)
    assert message == (
        "API key environment variable "
        f"'{PRIMARY_KEY}' contains a control character."
    )
    assert secret_value not in message
    assert "secret-prefix" not in message
    assert "secret-suffix" not in message


def test_invalid_selected_alias_does_not_fall_through_to_later_alias():
    invalid_value = "secret\nvalue"
    environment = TrackingEnvironment(
        {
            PRIMARY_KEY: "",
            FIRST_ALIAS: invalid_value,
            SECOND_ALIAS: "valid-later-key",
        }
    )

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            environment,
            _catalog(aliases=[FIRST_ALIAS, SECOND_ALIAS]),
        )

    assert str(error.value) == (
        "API key environment variable "
        f"'{FIRST_ALIAS}' contains a control character."
    )
    assert invalid_value not in str(error.value)
    assert "valid-later-key" not in str(error.value)
    assert environment.lookups == [PRIMARY_KEY, FIRST_ALIAS]


@pytest.mark.parametrize(
    ("declares_override", "override_value"),
    [
        (False, None),
        (True, None),
        (True, ""),
        (True, "\t\u2003\n"),
    ],
)
def test_static_url_is_preserved_byte_for_byte_without_populated_override(
    declares_override, override_value
):
    static_url = " HTTP://legacy host.invalid/unsupported/?query#fragment\\ "
    endpoint_variable = ENDPOINT_VARIABLE if declares_override else OMITTED
    environment = {PRIMARY_KEY: "test-key"}
    if override_value is not None:
        environment[ENDPOINT_VARIABLE] = override_value

    client = _client_with_environment(
        environment,
        _catalog(
            endpoint_variable=endpoint_variable,
            static_url=static_url,
        ),
    )

    assert client.api_url == static_url


@pytest.mark.parametrize(
    ("override_value", "expected_url"),
    [
        (
            "https://api.example.invalid/v1/chat/completions",
            "https://api.example.invalid/v1/chat/completions",
        ),
        (
            "https://api.example.invalid/v1",
            "https://api.example.invalid/v1/chat/completions",
        ),
        (
            " \t\nhttps://api.example.invalid/v1\r\v\f ",
            "https://api.example.invalid/v1/chat/completions",
        ),
        (
            "https://api.example.invalid/v1////",
            "https://api.example.invalid/v1/chat/completions",
        ),
        (
            "https://api.example.invalid/v1/chat/completions///",
            "https://api.example.invalid/v1/chat/completions",
        ),
        (
            "https://workspace.cn-beijing.example.invalid/compatible-mode/v1",
            "https://workspace.cn-beijing.example.invalid/compatible-mode/"
            "v1/chat/completions",
        ),
        (
            "https://regional.example.invalid/api/v1/chat/completions",
            "https://regional.example.invalid/api/v1/chat/completions",
        ),
        (
            "https://private-gateway.internal/openai/v1",
            "https://private-gateway.internal/openai/v1/chat/completions",
        ),
        (
            "https://192.0.2.1:8443/v1",
            "https://192.0.2.1:8443/v1/chat/completions",
        ),
        (
            "https://[2001:db8::1]:443/v1",
            "https://[2001:db8::1]:443/v1/chat/completions",
        ),
        (
            "HTTPS://EXAMPLE.invalid:1/prefix/v1",
            "HTTPS://EXAMPLE.invalid:1/prefix/v1/chat/completions",
        ),
        (
            "https://example.invalid:65535/gateway%3Fname/v1",
            "https://example.invalid:65535/gateway%3Fname/v1/chat/completions",
        ),
    ],
)
def test_endpoint_override_accepts_supported_shapes_and_preserves_bytes(
    override_value, expected_url
):
    client = _client_with_environment(
        {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
        _catalog(endpoint_variable=ENDPOINT_VARIABLE),
    )

    assert client.api_url == expected_url


@pytest.mark.parametrize(
    ("case_name", "override_value"),
    [
        ("http", "http://example.invalid/v1"),
        ("ftp", "ftp://example.invalid/v1"),
        ("relative", "/v1"),
        ("missing-authority", "https:/v1"),
        ("missing-host", "https:///v1"),
        ("empty-host", "https://:443/v1"),
        ("empty-port", "https://example.invalid:/v1"),
        ("alphabetic-port", "https://example.invalid:port/v1"),
        ("zero-port", "https://example.invalid:0/v1"),
        ("negative-port", "https://example.invalid:-1/v1"),
        ("out-of-range-port", "https://example.invalid:65536/v1"),
        ("userinfo-name", "https://user@example.invalid/v1"),
        ("userinfo-empty-password", "https://user:@example.invalid/v1"),
        ("userinfo-password", "https://:password@example.invalid/v1"),
        ("userinfo-empty", "https://@example.invalid/v1"),
        ("empty-query", "https://example.invalid/v1?"),
        ("query", "https://example.invalid/v1?token=value"),
        ("empty-fragment", "https://example.invalid/v1#"),
        ("fragment", "https://example.invalid/v1#fragment"),
        ("authority-backslash", "https://example.invalid\\attacker/v1"),
        ("path-backslash", "https://example.invalid/openai\\v1"),
        ("root-path", "https://example.invalid/"),
        ("no-path", "https://example.invalid"),
        ("v2-path", "https://example.invalid/v2"),
        ("v11-path", "https://example.invalid/v11"),
        ("models-path", "https://example.invalid/v1/models"),
        (
            "extra-after-completions",
            "https://example.invalid/v1/chat/completions/extra",
        ),
        ("invalid-ip-literal", "https://[not-an-ip]/v1"),
    ],
)
def test_endpoint_override_rejects_invalid_scheme_authority_port_and_path(
    case_name, override_value
):
    with pytest.raises(ValueError) as error:
        _client_with_environment(
            {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
            _catalog(endpoint_variable=ENDPOINT_VARIABLE),
        )

    assert ENDPOINT_VARIABLE in str(error.value), case_name
    assert override_value not in str(error.value), case_name


@pytest.mark.parametrize("control_code", [*range(33), 127])
def test_endpoint_override_rejects_internal_ascii_control_and_space(
    control_code,
):
    override_value = (
        "https://example.invalid/prefix" f"{chr(control_code)}suffix/v1"
    )

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
            _catalog(endpoint_variable=ENDPOINT_VARIABLE),
        )

    assert ENDPOINT_VARIABLE in str(error.value)
    assert override_value not in str(error.value)


@pytest.mark.parametrize(
    "unicode_whitespace",
    ["\u0085", "\u00a0", "\u1680", "\u2003", "\u2028", "\u2029", "\u3000"],
)
def test_endpoint_override_rejects_remaining_unicode_whitespace(
    unicode_whitespace,
):
    override_value = (
        "https://example.invalid/prefix" f"{unicode_whitespace}suffix/v1"
    )

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
            _catalog(endpoint_variable=ENDPOINT_VARIABLE),
        )

    assert ENDPOINT_VARIABLE in str(error.value)
    assert override_value not in str(error.value)


def test_invalid_endpoint_error_leaks_no_value_or_parsed_component():
    override_value = (
        "http://secret-host.example.invalid:8080/private?token=secret-value"
    )

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
            _catalog(endpoint_variable=ENDPOINT_VARIABLE),
        )

    message = str(error.value)
    assert ENDPOINT_VARIABLE in message
    for forbidden in (
        override_value,
        "secret-host",
        "8080",
        "private",
        "secret-value",
    ):
        assert forbidden not in message


def test_url_parser_failure_does_not_chain_component_bearing_exception():
    override_value = "https://[secret-host]/v1"

    with pytest.raises(ValueError) as error:
        _client_with_environment(
            {PRIMARY_KEY: "test-key", ENDPOINT_VARIABLE: override_value},
            _catalog(endpoint_variable=ENDPOINT_VARIABLE),
        )

    assert "secret-host" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_invalid_endpoint_fails_before_http_client_or_request_construction():
    with patch("llm_exec_core.client.httpx.AsyncClient") as async_client:
        with patch("llm_exec_core.client.httpx.Request") as request:
            with pytest.raises(ValueError):
                _client_with_environment(
                    {
                        PRIMARY_KEY: "test-key",
                        ENDPOINT_VARIABLE: "http://example.invalid/v1",
                    },
                    _catalog(endpoint_variable=ENDPOINT_VARIABLE),
                )

    async_client.assert_not_called()
    request.assert_not_called()


def test_client_resolution_does_not_mutate_caller_owned_settings():
    catalog = _catalog(
        aliases=[FIRST_ALIAS, SECOND_ALIAS],
        endpoint_variable=ENDPOINT_VARIABLE,
    )
    original = deepcopy(catalog)

    first = _client_with_environment(
        {
            PRIMARY_KEY: "",
            FIRST_ALIAS: "alias-key",
            ENDPOINT_VARIABLE: "https://override.example.invalid/v1",
        },
        catalog,
    )
    second = _client_with_environment(
        {PRIMARY_KEY: "primary-key"},
        catalog,
    )

    assert first.api_url == (
        "https://override.example.invalid/v1/chat/completions"
    )
    assert second.api_url == STATIC_URL
    assert catalog == original
