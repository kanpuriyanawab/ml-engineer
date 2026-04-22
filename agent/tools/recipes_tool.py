"""
Recipe discovery tools — agent-internal read-only tools for finding and reading
fine-tuning recipes. Users never invoke these by name; the agent calls them
automatically when figuring out how to fulfill a fine-tuning request.

Each recipe YAML encodes a (framework × runtime) combination with a complete,
runnable training script template.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

RECIPES_DIR = Path(__file__).parent.parent / "recipes"

LIST_RECIPES_TOOL_SPEC = {
    "name": "list_recipes",
    "description": (
        "List available fine-tuning recipes. Call this when the user wants to fine-tune "
        "a model and you need to pick the right (framework × runtime) combination. "
        "Each recipe is a complete, tested training script template — use recipes "
        "instead of writing training code from scratch. "
        "Recipes are organized by runtime (local/modal/hf_jobs) and framework "
        "(mlx-lm/unsloth/trl). Filter by runtime or framework to narrow results."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "runtime": {
                "type": "string",
                "enum": ["local", "modal", "hf_jobs", "any"],
                "description": "Filter by runtime. Default: 'any'.",
            },
            "framework": {
                "type": "string",
                "enum": ["mlx-lm", "unsloth", "trl", "transformers", "any"],
                "description": "Filter by training framework. Default: 'any'.",
            },
        },
    },
}

GET_RECIPE_TOOL_SPEC = {
    "name": "get_recipe",
    "description": (
        "Get the full details of a fine-tuning recipe by name, including the complete "
        "training script template, all required/optional parameters, and dependencies. "
        "For local/hf_jobs recipes: extract the script and dependencies from the YAML, "
        "substitute {{param}} placeholders with actual values, then pass to "
        "local_finetune or hf_jobs as script= and dependencies=. "
        "For modal recipes: pass recipe= and recipe_params= to modal_finetune — "
        "the tool handles script rendering and Modal wrapping automatically."
    ),
    "parameters": {
        "type": "object",
        "required": ["name"],
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "description": "Recipe name as returned by list_recipes (e.g. 'unsloth-modal', 'mlx-lora-mac').",
            },
        },
    },
}


def _load_recipe(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_recipe_summary(recipe: dict) -> str:
    name = recipe.get("name", "unknown")
    framework = recipe.get("framework", "?")
    runtime = recipe.get("runtime", "?")
    description = recipe.get("description", "").strip()
    size_range = recipe.get("recommended_model_size_range", "any")
    gpu = recipe.get("gpu", recipe.get("hardware_flavor", ""))
    cost = recipe.get("estimated_cost_per_hour", "")
    params_info = recipe.get("params", {})
    required = [p["name"] for p in params_info.get("required", [])]

    lines = [
        f"name: {name}",
        f"framework: {framework} | runtime: {runtime}",
        f"model_size: {size_range}",
    ]
    if gpu:
        lines.append(f"gpu: {gpu}")
    if cost:
        lines.append(f"cost: ~${cost}/hr")
    if required:
        lines.append(f"required_params: {', '.join(required)}")
    lines.append(f"description: {description[:120]}{'...' if len(description) > 120 else ''}")
    return "\n".join(lines)


async def list_recipes_handler(
    args: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    if not RECIPES_DIR.exists():
        return "No recipes directory found.", False

    runtime_filter = args.get("runtime", "any")
    framework_filter = args.get("framework", "any")

    recipes = []
    for path in sorted(RECIPES_DIR.glob("*.yaml")):
        try:
            recipe = _load_recipe(path)
        except Exception as e:
            continue

        if runtime_filter != "any" and recipe.get("runtime") != runtime_filter:
            continue
        if framework_filter != "any" and recipe.get("framework") != framework_filter:
            continue

        recipes.append(recipe)

    if not recipes:
        return f"No recipes found (runtime={runtime_filter}, framework={framework_filter}).", True

    sections = []
    for recipe in recipes:
        sections.append(_format_recipe_summary(recipe))

    header = f"Found {len(recipes)} recipe(s)"
    if runtime_filter != "any":
        header += f" [runtime={runtime_filter}]"
    if framework_filter != "any":
        header += f" [framework={framework_filter}]"

    return header + "\n\n" + "\n\n---\n\n".join(sections), True


async def get_recipe_handler(
    args: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    name = args.get("name", "").strip()
    if not name:
        return "ERROR: 'name' is required.", False

    if not RECIPES_DIR.exists():
        return "No recipes directory found.", False

    path = RECIPES_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in sorted(RECIPES_DIR.glob("*.yaml"))]
        return (
            f"Recipe '{name}' not found. Available: {', '.join(available)}",
            False,
        )

    try:
        recipe = _load_recipe(path)
    except Exception as e:
        return f"Failed to load recipe '{name}': {e}", False

    # Return full YAML as text so the agent can read script template + all params
    import yaml as _yaml
    content = _yaml.dump(recipe, default_flow_style=False, allow_unicode=True, width=120)
    return content, True
