"""Run the simulator until the ordinary shop template pool is covered and verified."""

import json

from harness.live import Live, main
from harness.screen.read import CATALOG, ROOT, SHOP_PORTRAITS
from tools.audit_shop_portraits import audit_templates, shop_cards


class Collector(Live):
    def __init__(self, args):
        if args.mode != "auto" or args.stop_at_leader_prep:
            raise ValueError("Collection requires auto mode without a preparation stop")
        self.pool = shop_cards(CATALOG)
        if len(self.pool) != 202:
            raise ValueError(f"Expected 202 ordinary shop IDs, catalog has {len(self.pool)}")
        self.present = -1
        super().__init__(args)

    def missing_templates(self):
        return self.pool.keys() - {path.stem for path in SHOP_PORTRAITS.glob("*.png")}

    def opening_bond_choice(self, screen, highlighted):
        default = super().opening_bond_choice(screen, highlighted)
        match = screen["match"]
        groups = match["bondChoice"]["groups"]
        disabled = set(match.get("disabledBonds") or []) - set(groups[0]) - set(groups[1])
        choices = [sorted(disabled | set(groups[1 - index])) for index in range(2)]
        missing = self.missing_templates() & set(self.rules.visible)
        iberia = {key for key in missing if self.pool[key].get("nationId") == "iberia"}
        available = [missing - self.rules.banned_at_start(off) for off in choices]
        index = max(range(2), key=lambda i: (len(iberia & available[i]), len(available[i]), i == default["group"]))
        if index == default["group"]:
            return default
        bond = next((bond for bond in self.rules.plan.order
                     if bond in groups[index] and not self.rules.bond_unavailable(bond, choices[index])), None)
        return {"group": index, "bond": bond, "disabledBonds": choices[index]}

    def report_coverage(self, *, verify=False):
        report = audit_templates(self.pool, verify=verify)
        path = ROOT / "debug/shop-portrait-audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)
        if report["present"] != self.present:
            self.present = report["present"]
            print(f"Template coverage {self.present}/{len(self.pool)}", flush=True)
        return report

    def prep(self):
        present = len(self.pool) - len(self.missing_templates())
        if present == len(self.pool):
            report = self.report_coverage(verify=True)
            if report["loadFailed"]:
                raise ValueError(f"MFW could not load templates {report['loadFailed']}")
            if report["loaded"] == len(self.pool):
                print(f"MFW verified all {report['loaded']} shop templates", flush=True)
                return True
        elif present != self.present:
            self.report_coverage()
        return super().prep()

    def run(self, games):
        """Continue across match ends. Allow one fresh re-observation per failed round."""
        retried = None
        while True:
            status = super().run(None)
            if status != 3:
                if status != 0:
                    self.report_coverage(verify=True)
                return status
            checkpoint = (self.state.game, self.state.round)
            if checkpoint == retried or self.observe() not in {
                "preparation", "battle", "settlement", "disconnect", "matchEnded",
            }:
                self.report_coverage(verify=True)
                return status
            retried = checkpoint
            self.a.resume = False
            self._retries.clear()
            self._shop_open = self._prep_chain = False
            print(f"Collection recheck at game {checkpoint[0]} round {checkpoint[1]}", flush=True)


if __name__ == "__main__":
    main(Collector)
