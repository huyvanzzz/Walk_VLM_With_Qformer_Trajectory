import json

import requests

from gptscore.constants import (
    CRITERION_KEYS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    SCHEMA_VERSION,
)
from gptscore.env_utils import get_env_value, load_runtime_env


SYSTEM_PROMPT = (
    "You are a careful evaluator for spoken navigation guidance for a visually impaired user. "
    "Score only the final answer quality using the provided JSON schema."
)

USER_PROMPT_TEMPLATE = """Ground truth:
{{GROUND_TRUTH}}

Model generation:
{{GENERATION}}

Evaluate the generation against the ground truth using the required criteria only. Mark a criterion as not applicable only when it truly cannot be judged from the pair. Keep rationales concise."""


def _criterion_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "applicable": {"type": "boolean"},
            "label": {"type": ["string", "null"], "enum": ["Fail", "Weak", "Acceptable", "Strong", None]},
            "rationale": {"type": "string"},
        },
        "required": ["applicable", "label", "rationale"],
    }


def build_json_schema():
    criteria_properties = {key: _criterion_schema() for key in CRITERION_KEYS}
    return {
        "name": SCHEMA_VERSION,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "criteria": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": criteria_properties,
                    "required": CRITERION_KEYS,
                },
                "overall_rationale": {"type": "string"},
            },
            "required": ["criteria", "overall_rationale"],
        },
    }


def resolve_provider_config(provider, model=None):
    if provider == "openrouter":
        return {
            "provider": "openrouter",
            "model": model or DEFAULT_OPENROUTER_MODEL,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

    return {
        "provider": "openai",
        "model": model or DEFAULT_OPENAI_MODEL,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    }


def render_user_prompt(template, ground_truth, generation):
    return (
        template.replace("{{GROUND_TRUTH}}", ground_truth).replace(
            "{{GENERATION}}", generation
        )
    )


def build_chat_completion_payload(model, system_prompt, user_prompt, json_schema):
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }


def make_provider_judge_callable(
    provider,
    model,
    repo_root,
    max_retries=2,
    timeout=90,
):
    loaded_env = load_runtime_env(repo_root)
    provider_config = resolve_provider_config(provider, model)
    api_key = get_env_value(provider_config["api_key_env"], loaded_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key: {provider_config['api_key_env']}. Export the variable or provide it through your runtime environment."
        )

    json_schema = build_json_schema()
    url = provider_config["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = get_env_value("OPENROUTER_HTTP_REFERER", loaded_env) or "http://localhost"
        headers["X-Title"] = get_env_value("OPENROUTER_X_TITLE", loaded_env) or "vlm-gptscore"

    def call_once(ground_truth, generation):
        user_prompt = render_user_prompt(USER_PROMPT_TEMPLATE, ground_truth, generation)
        payload = build_chat_completion_payload(
            provider_config["model"], SYSTEM_PROMPT, user_prompt, json_schema
        )
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        raw_text = response.text
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {raw_text[:500]}")

        body = response.json()
        choice = body["choices"][0]["message"]
        if "refusal" in choice and choice["refusal"]:
            raise RuntimeError(f"Model refusal: {choice['refusal']}")
        content = choice.get("content")
        if not isinstance(content, str):
            raise ValueError("Model content is not a plain JSON string")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON content: {exc}") from exc
        return parsed, raw_text

    def judge_callable(ground_truth, generation):
        attempts = 0
        last_error = None
        while attempts <= max_retries:
            try:
                return call_once(ground_truth, generation)
            except RuntimeError as exc:
                attempts += 1
                last_error = exc
                if attempts > max_retries:
                    raise
            except ValueError as exc:
                attempts += 1
                last_error = exc
                if attempts > max_retries:
                    raise
        raise last_error

    return judge_callable, provider_config["model"]
