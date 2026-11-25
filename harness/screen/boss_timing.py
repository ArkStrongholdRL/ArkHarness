"""Collect the countdown and wall-clock endpoints on existing MFW battle frames."""

import time

from .native import Frame
from .read import screen_integer


class BossTiming:
    def __init__(self, data: dict, report):
        self.data = data
        self.report = report

    def timer(self, frame: Frame, now: float) -> int | None:
        value = screen_integer(frame, "Battle.LeaderTimer")
        if value is None or not 0 <= value <= 180:
            return None
        previous = self.data.get("lastTimer")
        if previous is not None:
            elapsed = now - self.data["lastTimerAt"]
            if value > previous or previous - value > max(0, elapsed) * 3 + 2:
                return None
        self.data.update(lastTimer=value, lastTimerAt=now)
        return value

    def start(self, frame: Frame) -> None:
        if "startedAt" in self.data:
            return
        now = time.time()
        self.data.update(startedAt=now, startTimer=self.timer(frame, now))
        self.report("start")

    def sample(self, frame: Frame) -> None:
        now = time.time()
        ended = self.data.get("endedAt")
        if ended is not None and (self.data.get("outcome") or now - ended > 4):
            return
        if not frame.hit("Battle.LeaderHud"):
            return
        if ended is not None:
            if any(row.text.startswith("完美作战") for row in frame.rows("Battle.ResultText")):
                self.end(now, None, None, "settlement", "cleared")
            return
        value = self.timer(frame, now)
        life = screen_integer(frame, "Screen.Life")
        if life is not None:
            self.data.update(lastLife=life, lastLifeAt=now)
        if value == 0 and "zeroAt" not in self.data:
            self.data["zeroAt"] = now
            self.report("zero")
        texts = {row.text.rstrip("！!") for row in frame.rows("Battle.ResultText")}
        if texts & {"作战结束", "完美作战"}:
            outcome = "cleared" if "完美作战" in texts else "failed" if life == 0 else None
            self.end(now, value, life, "settlement", outcome)

    def end(self, now: float, value: int | None, life: int | None, source: str, outcome: str | None) -> None:
        ended = "endedAt" in self.data
        if not ended:
            self.data.update(endedAt=now, endSource=source, endTimer=value, endLife=life)
        if outcome is not None and outcome != self.data.get("outcome"):
            self.data["outcome"] = outcome
        elif ended:
            return
        self.report("end")

    def finish(self, snapshot: dict) -> None:
        screen = snapshot["screen"]
        outcome = None
        if screen == "matchInterrupted":
            passed = (snapshot.get("result") or {}).get("passedRounds")
            if passed is not None:
                outcome = "cleared" if passed >= self.data["round"] else "failed"
        elif snapshot.get("round") is not None and snapshot["round"] > self.data["round"]:
            outcome = "cleared"
        elif screen in {"bountySelection", "equipmentSelection", "tacticSelection"}:
            outcome = "cleared"
        else:
            return
        self.end(time.time(), None, None, screen, outcome)
