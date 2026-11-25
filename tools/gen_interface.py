"""Generate the MaaDebugger interface from selected pipeline entries."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "resource/pipeline"
OUTPUT = ROOT / "interface.json"

HEADER = {
    "interface_version": 2,
    "name": "ArkHarness",
    "label": "ArkHarness 卫戍协议读屏",
    "version": "v0.1.0",
    "description": ("ArkHarness 的 MaaFramework 资源包，仅供调试（MaaDebugger 等）查看节点识别结果。"
                    "harness 本身在 Python 里加载 resource/ 并注册自定义识别，不经过本文件。"
                    "基准分辨率 1280×720（原始尺寸截图）。"),
    "controller": [{"name": "MuMu-Adb", "label": "MuMu 12（ADB，1280×720 原始截图）",
                    "type": "Adb", "display_raw": True}],
    "resource": [{"name": "ArkHarness", "label": "ArkHarness", "path": ["./resource"]}],
    "group": [
        {"name": "probe", "label": "纯识别（可单独运行）", "default_expand": True},
        {"name": "flow", "label": "流程入口（依赖 Python 自定义识别）", "default_expand": False},
    ],
}

FLOW = (
    ("ReadScreen", "读屏：输出 ScreenState", "ArkReadScreen"),
    ("ArkObserve", "导航：观察当前界面", "ArkNavigation"),
    ("ArkClosePanel", "导航：关闭面板回到准备界面", "ArkNavigation"),
    ("ArkInspectMatchInfo", "导航：打开本局信息", "ArkNavigation"),
    ("ArkInspectEnemies", "导航：打开敌人预告", "ArkNavigation"),
    ("ArkInspectBond", "导航：打开盟约详情", "ArkNavigation（参数：盟约名）"),
    ("ArkStartBattle", "导航：准备就绪并核对开战", None),
    ("Entry.EnterLayout", "入口：从卫戍协议入口进入布阵", "ArkNavigation；会点击"),
    ("Commander.Observe", "指挥官：读选择页", "ArkCommander"),
    ("Commander.Preview", "指挥官：预览某格", "ArkCommander（参数：行列）；会点击"),
    ("Commander.Confirm", "指挥官：确认", "ArkCommander（参数：名字）；会点击"),
    ("Shop.Buy", "商店：购买一格", "ArkShopBuy（参数：引擎槽位）；会买东西"),
    ("Board.Deploy", "棋盘：拖放一个干员", "ArkDeploy（参数：手牌格、格子、朝向）；会拖动"),
)

PROBES = (
    "Screen.Drag", "Screen.StageM06", "Screen.StageM04", "Screen.DeployExpanded", "Screen.DeployCollapsed",
    "Screen.BondPanel", "Screen.BondUnderline", "Screen.BondName", "Screen.BondLayers",
    "Screen.BondDetailName", "Screen.BondDetailLayers", "Screen.BondMembers",
    "Commander.GridPlain", "Commander.GridWithDetail", "Commander.Portrait",
    "Commander.SelectedName", "Commander.SelectedBadge",
    "Board.FinishFacing", "Board.FacingCancelText", "Board.CarriedName", "Board.SellText",
    "Board.LegalTile", "Board.FacingMarker",
    "Screen.EnemyPanel", "Screen.EnemyPreviewButton", "Screen.EnemyNames", "Lifecycle.Title",
    "Screen.MatchInfoButton", "Screen.MatchInfoCross", "Screen.EnabledBond",
    "ArkRoundNumber", "Screen.ReadyIcon", "Screen.ReadyColor", "Screen.Mode", "Screen.Life", "Screen.Funds",
    "Screen.FullText", "Shop.ConfirmBuyText", "Shop.CollapseText", "Shop.Expanded", "Screen.ShopButtons",
    "Screen.ShopLevelExpanded", "Screen.ShopLevelCollapsed", "Screen.UpgradePrice", "Screen.RefreshPrice",
    "Screen.ShopCardName", "Screen.ShopCardTier", "Screen.ShopCardPrice", "Screen.ShopSold",
)

# These entries predate the current pipeline and remain in the generated interface for parity.
MISSING_LEGACY_PROBES = {"Board.FinishFacing": "DEPLOY.json", "Board.LegalTile": "DEPLOY.json"}
PROBE_DESCRIPTION = "纯识别节点（不点击），可在调试器里对当前画面单独运行。"
FLOW_DESCRIPTION = "自定义识别只在 harness 的 Python 进程里注册，调试器里单独运行会在该节点失败；用于查看前面节点的识别框。"
READY_DESCRIPTION = "需要 Harness 注册 ArkNavigation 识别和 ArkRememberAndClickReady 动作。会点击「准备就绪」。"


def generate() -> str:
    nodes = {}
    for path in sorted(PIPELINE.glob("*.json")):
        for name, node in json.loads(path.read_text(encoding="utf-8")).items():
            if name in nodes:
                raise ValueError(f"Duplicate pipeline node: {name}")
            nodes[name] = path.name, node

    tasks = []
    for name, label, requirements in FLOW:
        source, _ = nodes[name]
        description = READY_DESCRIPTION if requirements is None else f"需要 {requirements}。{FLOW_DESCRIPTION}"
        tasks.append({"name": name, "label": label, "entry": name, "group": ["flow"],
                      "description": f"{description}来源：resource/pipeline/{source}"})
    for name in PROBES:
        if name not in nodes and name in MISSING_LEGACY_PROBES:
            source = MISSING_LEGACY_PROBES[name]
            print(f"Legacy probe has no pipeline node: {name}", file=sys.stderr)
        else:
            source, node = nodes[name]
            if (node.get("action") not in (None, "DoNothing") or node.get("custom_recognition")
                    or node.get("next") or node.get("on_error")):
                raise ValueError(f"Debugger probe is not a standalone recognition node: {name}")
        tasks.append({"name": name, "entry": name, "group": ["probe"],
                      "description": f"{PROBE_DESCRIPTION}来源：resource/pipeline/{source}"})
    return json.dumps({**HEADER, "task": tasks}, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check interface.json without writing")
    args = parser.parse_args()
    try:
        generated = generate()
        if args.check:
            if OUTPUT.read_text(encoding="utf-8") != generated:
                parser.exit(1, "interface.json differs. Run tools/gen_interface.py to regenerate it.\n")
        else:
            OUTPUT.write_text(generated, encoding="utf-8", newline="\n")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
