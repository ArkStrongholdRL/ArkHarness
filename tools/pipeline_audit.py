"""Static audit of resource/pipeline: defined nodes, references from JSON and Python, dead nodes."""
import ast, json, re, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
defs = {}
for f in sorted(glob.glob(str(ROOT / "resource/pipeline/*.json"))):
    for k, v in json.loads(open(f, encoding="utf-8").read()).items():
        if k in defs: print("DUPLICATE", k, f)
        defs[k] = (Path(f).name, v)
refs = {k: set() for k in defs}
def walk(v, owner):
    if isinstance(v, str):
        if v in defs and v != owner: refs[v].add("json:" + owner)
    elif isinstance(v, list):
        for x in v: walk(x, owner)
    elif isinstance(v, dict):
        for x in v.values(): walk(x, owner)
for k, (f, v) in defs.items():
    for key, val in v.items():
        if key in ("next", "on_error", "all_of", "any_of", "target", "anchor", "template") or isinstance(val, (list, dict)):
            walk(val, k)
for f in list((ROOT / "harness").rglob("*.py")) + list((ROOT / "tools").glob("*.py")):
    txt = f.read_text(encoding="utf-8", errors="ignore")
    for name in defs:
        if re.search(r'["\']' + re.escape(name) + r'["\']', txt): refs[name].add("py:" + f.name)
dead = [k for k in defs if not refs[k]]
print("nodes", len(defs), "unreferenced", len(dead))
for k in dead: print(f"  {defs[k][0]:22s} {k}")
if "--images" in sys.argv:
    used = set()
    for k, (f, v) in defs.items():
        for m in re.findall(r'"template":\s*(\[[^\]]*\]|"[^"]*")', json.dumps(v)): used |= set(re.findall(r'[\w.\-]+\.png', m))
    for img in sorted((ROOT / "resource/image").glob("*")):
        if img.name not in used: print("  unused image", img.name)

# reachability from Python entry points (+ the f-string families), and identical definitions
roots = {k for k, r in refs.items() if any(x.startswith("py:") for x in r)}
roots |= {k for k in defs if re.match(r"Bounty\.(Name|Description|Confirm)(Left|Center|Right)$", k)}
seen, todo = set(), list(roots)
while todo:
    k = todo.pop()
    if k in seen: continue
    seen.add(k)
    for other, r in refs.items():
        if "json:" + k in r and other not in seen: todo.append(other)
print("unreachable from any Python entry:", sorted(set(defs) - seen))
by = {}
for k, (f, v) in defs.items(): by.setdefault(json.dumps(v, sort_keys=True), []).append(k)
print("identical definitions:", [v for v in by.values() if len(v) > 1])
print("nodes with a custom recognition but no Python registration:")
session = ast.parse((ROOT / "harness/screen/session.py").read_text(encoding="utf-8"))
registry = next(node.value for node in session.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SCREEN_RECOGNIZERS"
                        for target in node.targets))
reg = set(ast.literal_eval(registry))
missing = []
for k, (f, v) in defs.items():
    c = v.get("custom_recognition")
    if c and c not in reg:
        missing.append((k, c))
for k, c in missing:
    print("  ", k, c)
if missing:
    sys.exit(1)
