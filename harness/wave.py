"""Resolve the deterministic template and route set for a normal solo round."""

import json
from pathlib import Path


WAVE_TABLES = Path(__file__).resolve().parents[1] / "data" / "wave_tables.json"


def resolve_leader_names(gd, round_number, kinds, ambiguous, boss_id):
    """Resolve an OCR name shared by ordinary and hidden leaders using the match round."""
    if not ambiguous or round_number is None or round_number not in (gd.boss_round, gd.hidden_round):
        return kinds, ambiguous, boss_id
    hidden = round_number == gd.hidden_round
    bosses = gd.raw.get("bosses") or {}
    leader_keys = {boss["enemyKey"] for boss in bosses.values()
                   if boss.get("enemyKey") and type(boss.get("hidden")) is bool}
    matches = []
    for label, candidates in ambiguous.items():
        candidate_keys = set(candidates)
        if not candidate_keys or not candidate_keys <= leader_keys:
            continue
        for key, boss in bosses.items():
            if (boss.get("hidden") is hidden and boss.get("enemyKey") in candidate_keys
                    and boss_id in (None, key)):
                matches.append((label, key, boss["enemyKey"]))
    if len(matches) != 1:
        return kinds, ambiguous, boss_id
    label, key, enemy_key = matches[0]
    return sorted({*kinds, enemy_key}), {name: ids for name, ids in ambiguous.items() if name != label}, key


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
