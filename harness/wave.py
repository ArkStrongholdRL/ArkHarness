"""Resolve the deterministic template and route set for a normal solo round."""

import json
from pathlib import Path


WAVE_TABLES = Path(__file__).resolve().parents[1] / "data" / "wave_tables.json"


def match_preview(state: dict, stage_id: str | None = None, round_number: int | None = None,
                  *, tables_path: Path = WAVE_TABLES) -> dict:
    """Attach template context to a preview while leaving the observed rows untouched."""
    match = state["match"]
    config = state["config"]
    observed_stage = match.get("stageId")
    observed_round = match.get("round")
    result = {"stageId": stage_id or observed_stage,
              "round": round_number if round_number is not None else observed_round,
              "templateId": None, "routes": None}

    if (stage_id and observed_stage and stage_id != observed_stage
            or round_number is not None and observed_round is not None and round_number != observed_round):
        return {"status": "contextMismatch", **result}

    stage = result["stageId"]
    current_round = result["round"]
    if not isinstance(stage, str) or not stage or type(current_round) is not int or current_round < 1:
        return {"status": "missingContext", **result}
    if current_round in (config.get("bossRound"), config.get("hiddenRound")):
        return {"status": "leaderRound", **result}

    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    counts = tables["ambiguous"]
    if counts["template"] or counts["routes"]:
        raise ValueError("The wave table contains ambiguous templates or route sets")
    template = tables["template"].get(f"{stage}|{current_round}")
    routes = tables["routes"].get(f"{stage}|{template}") if template is not None else None
    if routes is None:
        return {"status": "unmapped", **result}
    return {**result, "status": "matched", "templateId": template, "routes": routes}
