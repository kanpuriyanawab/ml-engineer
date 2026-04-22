"""
Modal fine-tuning tool — runs training scripts on Modal cloud GPUs via CLI subprocess.

Pattern mirrors local_finetune_tool: agent (or recipe) provides a Python script
with Modal decorators, this tool executes it via `modal run --detach` and streams
logs back as tool_log events.

For common use cases, pass recipe= + recipe_params= and the tool renders the
complete Modal script automatically. For custom scripts, pass script= directly.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml

from agent.core.session import Event

RECIPES_DIR = Path(__file__).parent.parent / "recipes"

MODAL_FINETUNE_TOOL_SPEC = {
    "name": "modal_finetune",
    "description": (
        "Run fine-tuning jobs on Modal cloud GPUs (L40S, A100, H100). "
        "Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env. "
        "Use for models that don't fit locally, or when you need CUDA and Unsloth speedups. "
        "Pass recipe= to use a pre-built recipe (recommended for non-ML users), "
        "or script= for a custom Modal Python script. "
        "GPU sizing: ≤7B→l40s, 8-14B→l40s/a100-40gb, 15-30B→a100-80gb, 30B+→h100. "
        "Operations: run (submit job), cancel (stop job), ps (list jobs), logs (fetch logs)."
    ),
    "parameters": {
        "type": "object",
        "required": ["operation"],
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "cancel", "ps", "logs"],
                "description": "run: submit job; cancel: stop it; ps: list active apps; logs: fetch logs by app_id",
            },
            "recipe": {
                "type": "string",
                "description": (
                    "Recipe name from list_recipes (e.g. 'unsloth-modal', 'trl-modal'). "
                    "Mutually exclusive with script. Tool loads and renders the recipe script."
                ),
            },
            "recipe_params": {
                "type": "object",
                "description": (
                    "Key-value overrides substituted into the recipe's {{param}} placeholders "
                    "(e.g. {\"model_id\": \"unsloth/Qwen3-14B\", \"output_repo\": \"user/model\"})."
                ),
                "additionalProperties": {"type": "string"},
            },
            "script": {
                "type": "string",
                "description": (
                    "Custom Python Modal script as a string. Must include modal.App, "
                    "@app.function, and @app.local_entrypoint. "
                    "Mutually exclusive with recipe."
                ),
            },
            "gpu": {
                "type": "string",
                "enum": ["l40s", "a10g", "a100-40gb", "a100-80gb", "h100"],
                "description": "GPU type override. If not set, uses recipe default or 'l40s'.",
            },
            "gpu_count": {
                "type": "integer",
                "description": "Number of GPUs. Default: 1.",
            },
            "timeout_hours": {
                "type": "integer",
                "description": "Max job runtime in hours. Default: 6.",
            },
            "app_id": {
                "type": "string",
                "description": "Modal app ID (ap-...). Required for: cancel, logs.",
            },
            "output_repo": {
                "type": "string",
                "description": "HF repo to push trained model to. Sets HF_OUTPUT_REPO env var.",
            },
        },
    },
}


def _check_modal_auth() -> tuple[str | None, str | None]:
    return os.environ.get("MODAL_TOKEN_ID"), os.environ.get("MODAL_TOKEN_SECRET")


def _render_recipe_script(recipe: dict, recipe_params: dict, gpu: str, timeout_hours: int) -> tuple[str, bool]:
    """Render a recipe's script template by substituting {{param}} placeholders."""
    script = recipe.get("script", "")
    if not script:
        return "ERROR: Recipe has no 'script' field.", False

    # Build substitution map from recipe defaults + caller overrides
    params_spec = recipe.get("params", {})
    subs: dict[str, str] = {}

    # Collect defaults from recipe
    for p in params_spec.get("optional", []):
        subs[p["name"]] = str(p.get("default", ""))

    # Override with recipe defaults for runtime/gpu
    subs["gpu"] = recipe.get("gpu", gpu)
    subs["timeout_hours"] = str(recipe.get("timeout_hours", timeout_hours))

    # Apply caller-provided recipe_params (highest priority)
    for k, v in (recipe_params or {}).items():
        subs[k] = str(v)

    # Substitute {{param}} placeholders
    def substitute(m: re.Match) -> str:
        key = m.group(1)
        if key not in subs:
            return m.group(0)  # leave unresolved placeholders as-is
        return subs[key]

    rendered = re.sub(r"\{\{(\w+)\}\}", substitute, script)

    # Check required params are all resolved
    missing = []
    for p in params_spec.get("required", []):
        name = p["name"]
        if name not in subs or not subs[name]:
            missing.append(name)
    if missing:
        return (
            f"ERROR: Missing required recipe params: {', '.join(missing)}. "
            f"Pass them via recipe_params.",
            False,
        )

    return rendered, True


async def _run_modal(args: dict[str, Any], session: Any) -> tuple[str, bool]:
    token_id, token_secret = _check_modal_auth()
    if not token_id or not token_secret:
        return (
            "ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in .env. "
            "Get them at https://modal.com/settings/tokens",
            False,
        )

    recipe_name = args.get("recipe", "").strip()
    script = args.get("script", "").strip()
    recipe_params = args.get("recipe_params") or {}
    gpu = args.get("gpu") or "l40s"
    timeout_hours = int(args.get("timeout_hours") or 6)

    if recipe_name and script:
        return "ERROR: Provide either 'recipe' or 'script', not both.", False

    if recipe_name:
        recipe_path = RECIPES_DIR / f"{recipe_name}.yaml"
        if not recipe_path.exists():
            available = [p.stem for p in sorted(RECIPES_DIR.glob("*.yaml"))]
            return (
                f"ERROR: Recipe '{recipe_name}' not found. Available: {', '.join(available)}",
                False,
            )
        try:
            with open(recipe_path, encoding="utf-8") as f:
                recipe = yaml.safe_load(f)
        except Exception as e:
            return f"ERROR: Failed to load recipe '{recipe_name}': {e}", False

        script, ok = _render_recipe_script(recipe, recipe_params, gpu, timeout_hours)
        if not ok:
            return script, False

    if not script:
        return "ERROR: Provide 'recipe' or 'script' for operation='run'.", False

    # Write rendered script to temp file
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"modal_job_{uuid.uuid4().hex[:8]}.py"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(script)

        env = {
            **os.environ,
            "MODAL_TOKEN_ID": token_id,
            "MODAL_TOKEN_SECRET": token_secret,
        }
        output_repo = args.get("output_repo", "")
        if output_repo:
            env["HF_OUTPUT_REPO"] = output_repo

        if session:
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "modal_finetune", "log": f"Submitting Modal job: {tmp_path}"},
            ))

        # Submit job with --detach
        submit_proc = await asyncio.create_subprocess_exec(
            "modal", "run", "--detach", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        submit_output_lines: list[str] = []
        assert submit_proc.stdout is not None
        async for raw in submit_proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            submit_output_lines.append(line)
            if session:
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "modal_finetune", "log": line},
                ))

        await submit_proc.wait()
        if submit_proc.returncode != 0:
            tail = "\n".join(submit_output_lines[-20:])
            return f"ERROR: Modal job submission failed (exit {submit_proc.returncode}).\n{tail}", False

        # Parse app_id from submission output
        submit_text = "\n".join(submit_output_lines)
        app_id_match = re.search(r"\b(ap-[a-zA-Z0-9]+)\b", submit_text)
        if not app_id_match:
            return (
                f"Job submitted but could not parse app_id from output.\n"
                f"Run `modal app list` to find your job.\n\nOutput:\n{submit_text}",
                True,
            )

        app_id = app_id_match.group(1)
        if session and hasattr(session, "_modal_app_ids"):
            session._modal_app_ids.add(app_id)

        if session:
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "modal_finetune", "log": f"Job started: {app_id} — streaming logs..."},
            ))

        # Stream logs via `modal app logs <app_id> --follow`
        log_proc = await asyncio.create_subprocess_exec(
            "modal", "app", "logs", app_id, "--follow",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        if session and hasattr(session, "_modal_app_ids"):
            # Store proc reference for potential cancel
            session._modal_log_procs = getattr(session, "_modal_log_procs", {})
            session._modal_log_procs[app_id] = log_proc

        log_lines: list[str] = []
        assert log_proc.stdout is not None
        async for raw in log_proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            log_lines.append(line)
            if session:
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "modal_finetune", "log": line},
                ))

        await log_proc.wait()

        if session and hasattr(session, "_modal_app_ids"):
            session._modal_app_ids.discard(app_id)

        summary = f"Modal job {app_id} completed."
        if output_repo:
            summary += f"\nModel pushed to: https://huggingface.co/{output_repo}"
        return summary, True

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _cancel_modal(args: dict[str, Any], session: Any) -> tuple[str, bool]:
    token_id, token_secret = _check_modal_auth()
    if not token_id or not token_secret:
        return "ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in .env.", False

    app_id = args.get("app_id", "").strip()
    if not app_id:
        running = list(getattr(session, "_modal_app_ids", set()))
        if not running:
            return "No active Modal apps found. Provide app_id explicitly.", False
        app_id = running[0]

    env = {
        **os.environ,
        "MODAL_TOKEN_ID": token_id,
        "MODAL_TOKEN_SECRET": token_secret,
    }
    proc = await asyncio.create_subprocess_exec(
        "modal", "app", "stop", app_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    out, _ = await proc.communicate()
    output = out.decode("utf-8", errors="replace").strip()

    if session and hasattr(session, "_modal_app_ids"):
        session._modal_app_ids.discard(app_id)

    if proc.returncode == 0:
        return f"Modal app {app_id} stopped.", True
    return f"Failed to stop {app_id} (exit {proc.returncode}).\n{output}", False


async def _ps_modal(args: dict[str, Any], session: Any) -> tuple[str, bool]:
    token_id, token_secret = _check_modal_auth()
    if not token_id or not token_secret:
        return "ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in .env.", False

    env = {
        **os.environ,
        "MODAL_TOKEN_ID": token_id,
        "MODAL_TOKEN_SECRET": token_secret,
    }
    proc = await asyncio.create_subprocess_exec(
        "modal", "app", "list",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    out, _ = await proc.communicate()
    output = out.decode("utf-8", errors="replace").strip()

    # Add session-tracked app IDs as context
    tracked = list(getattr(session, "_modal_app_ids", set()))
    suffix = f"\n\nSession-tracked app IDs: {', '.join(tracked)}" if tracked else ""
    return output + suffix, proc.returncode == 0


async def _logs_modal(args: dict[str, Any], session: Any) -> tuple[str, bool]:
    token_id, token_secret = _check_modal_auth()
    if not token_id or not token_secret:
        return "ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set in .env.", False

    app_id = args.get("app_id", "").strip()
    if not app_id:
        return "ERROR: 'app_id' is required for operation='logs'.", False

    env = {
        **os.environ,
        "MODAL_TOKEN_ID": token_id,
        "MODAL_TOKEN_SECRET": token_secret,
    }
    proc = await asyncio.create_subprocess_exec(
        "modal", "app", "logs", app_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    out, _ = await proc.communicate()
    output = out.decode("utf-8", errors="replace").strip()
    return output or f"No logs found for {app_id}.", proc.returncode == 0


async def modal_finetune_handler(
    args: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    operation = args.get("operation", "")
    if operation == "run":
        return await _run_modal(args, session)
    if operation == "cancel":
        return await _cancel_modal(args, session)
    if operation == "ps":
        return await _ps_modal(args, session)
    if operation == "logs":
        return await _logs_modal(args, session)
    return f"Unknown operation: {operation!r}. Use 'run', 'cancel', 'ps', or 'logs'.", False
