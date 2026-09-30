"""Portkey gateway integration and future-ready routing blueprints.

This Portkey workspace blocks inline ``x-portkey-config`` JSON. Production
requests therefore use saved config IDs (``pc-...``) when configured. The
``GATEWAY_CONFIG`` and feature templates below are source-controlled
blueprints: create or update the matching saved config in Portkey, then place
its ID in ``PORTKEY_CONFIG_ID`` or ``PORTKEY_GUARD_CONFIG_ID``.

Without a saved config ID, the client safely uses the selected Portkey Model
Catalog provider alias directly. That fallback is intentionally minimal: it
does not claim to enable gateway-side retries, routing, caching, or failover.
"""

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from hr_assistant import config
from hr_assistant.logger import get_logger


logger = get_logger(__name__)


# Kept as public names for existing imports. Values are Portkey Model Catalog
# provider aliases (for example, ``@hrpolicy``), despite the legacy env names.
PRIMARY_PROVIDER = config.PORTKEY_VIRTUAL_KEY
FALLBACK_PROVIDER = config.PORTKEY_VIRTUAL_BACKUP_KEY
GUARD_PROVIDER = config.PORTKEY_VIRTUAL_GUARD_KEY
JUDGE_PROVIDER = config.PORTKEY_VIRTUAL_JUDGE_KEY or PRIMARY_PROVIDER

@dataclass(frozen=True)
class GatewayRoute:
    """A named model route and the saved Portkey config that governs it."""

    name: str
    model: str
    provider_alias: str
    saved_config_id: str | None = None
    fallback_provider_alias: str | None = None


MAIN_ROUTE = GatewayRoute(
    name="assistant",
    model=config.LLM_MODEL_NAME,
    provider_alias=PRIMARY_PROVIDER,
    fallback_provider_alias=FALLBACK_PROVIDER,
    saved_config_id=config.PORTKEY_CONFIG_ID,
)
GUARD_ROUTE = GatewayRoute(
    name="guard",
    model=config.GUARD_MODEL_NAME,
    provider_alias=GUARD_PROVIDER,
    saved_config_id=config.PORTKEY_GUARD_CONFIG_ID,
)

JUDGE_ROUTE = GatewayRoute(
    name="judge",
    model=config.JUDGE_MODEL_NAME,
    provider_alias=JUDGE_PROVIDER,
    saved_config_id=config.PORTKEY_JUDGE_CONFIG_ID,
)

def _target(provider_alias: str, model_name: str) -> dict[str, Any]:
    """Return a Portkey config target backed by a Model Catalog alias."""
    return {
        "provider": provider_alias,
        "override_params": {"model": model_name},
    }


# Main assistant blueprint. Save this in Portkey to obtain PORTKEY_CONFIG_ID.
# It is never sent inline by this application.
GATEWAY_CONFIG: dict[str, Any] = {
    "retry": {"attempts": 2, 'on_status_codes': [429, 500, 503, 504]},
    "strategy": {"mode": "fallback"},
    "targets": [
        _target(PRIMARY_PROVIDER, config.LLM_MODEL_NAME),
        _target(FALLBACK_PROVIDER, config.LLM_MODEL_NAME),
    ],
}

# Guard blueprint. Save it separately because it must route only to the guard
# provider/model. It is never sent inline by this application.
GUARD_GATEWAY_CONFIG: dict[str, Any] = {
    "retry": {"attempts": 2, 'on_status_codes': [429, 500, 503, 504]},
    "targets": [_target(GUARD_PROVIDER, config.GUARD_MODEL_NAME)],
}

# Inactive fragments for the Portkey dashboard. Choose one routing strategy at
# a time: ``fallback``, ``loadbalance``, and ``conditional`` are alternatives.
# Review data/privacy implications before enabling either cache option.
PORTKEY_FEATURE_TEMPLATES: dict[str, dict[str, Any]] = {
    "simple_cache": {"cache": {"mode": "simple", "max_age": 300}},
    "semantic_cache": {"cache": {"mode": "semantic", "max_age": 300}},
    "load_balancing": {
        "strategy": {"mode": "loadbalance"},
        "targets": [
            {**_target(PRIMARY_PROVIDER, config.LLM_MODEL_NAME), "weight": 0.9},
            {**_target(FALLBACK_PROVIDER, config.LLM_MODEL_NAME), "weight": 0.1},
        ],
    },
    "conditional_routing": {
        "strategy": {
            "mode": "conditional",
            "conditions": [
                # {"query": {"metadata.user_tier": {"$eq": "premium"}},
                #  "then": "premium-model"},
            ],
            "default": "standard-model",
        },
        "targets": [
            {**_target(PRIMARY_PROVIDER, config.LLM_MODEL_NAME), "name": "standard-model"},
            {**_target(FALLBACK_PROVIDER, config.LLM_MODEL_NAME), "name": "premium-model"},
        ],
    },
    "guardrails": {
        "input_guardrails": ["guardrail-id-from-portkey"],
        "output_guardrails": ["guardrail-id-from-portkey"],
    },
}


def build_gateway_config(route: GatewayRoute) -> dict[str, Any]:
    """Return a fresh, editable blueprint for a Portkey saved configuration."""
    targets = [_target(route.provider_alias, route.model)]
    if route.fallback_provider_alias:
        targets.append(_target(route.fallback_provider_alias, route.model))

    gateway_config: dict[str, Any] = {"retry": {"attempts": 2, 'on_status_codes': [429, 500, 503, 504]}, "targets": targets}
    if route.fallback_provider_alias:
        gateway_config["strategy"] = {"mode": "fallback"}
    return gateway_config


def _headers_for_route(route: GatewayRoute) -> dict[str, str]:
    """Build headers without ever sending an inline Portkey config."""
    if route.saved_config_id:
        if not route.saved_config_id.startswith("pc-"):
            raise ValueError(f"{route.name} Portkey config ID must begin with pc-.")
        logger.info("Routing %s via saved Portkey config %s", route.name, route.saved_config_id)
        return createHeaders(api_key=config.PORTKEY_API_KEY, config=route.saved_config_id)

    logger.warning(
        "No saved Portkey config ID for %s; using provider alias %s directly "
        "without gateway-config features.",
        route.name,
        route.provider_alias,
    )
    return createHeaders(api_key=config.PORTKEY_API_KEY, provider=route.provider_alias)


def get_gateway_llm(
    model_name: str | None = None,
    *,
    primary_provider: str = PRIMARY_PROVIDER,
    fallback_provider: str | None = FALLBACK_PROVIDER,
    gateway_config_id: str | None = None,
    temperature: float | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> ChatOpenAI:
    """Return a Portkey-routed model.

    The optional arguments preserve specialized use cases such as the guard
    LLM. Passing a saved ``pc-...`` config enables the Portkey features that
    have been configured in the dashboard; otherwise direct provider routing
    is used without an inline config.
    """
    selected_model = model_name or config.LLM_MODEL_NAME
    route = GatewayRoute(
        name="custom" if model_name else MAIN_ROUTE.name,
        model=selected_model,
        provider_alias=primary_provider,
        fallback_provider_alias=fallback_provider,
        saved_config_id=gateway_config_id or (
            config.PORTKEY_CONFIG_ID if model_name is None else None
        ),
    )
    llm_options: dict[str, Any] = {
        "api_key": "portkey",
        "base_url": PORTKEY_GATEWAY_URL,
        "model": route.model,
        "default_headers": _headers_for_route(route),
        "timeout": config.PORTKEY_GATEWAY_TIMEOUT_SECONDS,
        "max_retries": config.PORTKEY_CLIENT_MAX_RETRIES,
    }
    if temperature is not None:
        llm_options["temperature"] = temperature
    if model_kwargs:
        llm_options["model_kwargs"] = model_kwargs
    return ChatOpenAI(**llm_options)



def get_gateway_judge_llm() -> ChatOpenAI:
    """Return the dedicated model used to judge evaluation results."""
    logger.info("Routing judge calls through Portkey (provider=%s)", JUDGE_PROVIDER)
    return get_gateway_llm(
        model_name=JUDGE_ROUTE.model,
        primary_provider=JUDGE_ROUTE.provider_alias,
        fallback_provider=None,
        gateway_config_id=JUDGE_ROUTE.saved_config_id,
    )
