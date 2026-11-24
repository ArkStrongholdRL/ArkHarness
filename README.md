# ArkHarness

卫戍协议：规程Harness。（还没做完捏）由 MaaFramework 强力驱动！
```

OCR 源目录需包含 `det.onnx`、`rec.onnx` 和 `keys.txt`。当前PP-OCRv6 small。引擎路径应指向含 `tools/rl/spec.mjs` 和 `tools/rl/py/model.py` 的训练版本，默认策略接口为 2570 维、100 个动作。

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m tools.test_soft_retry
& .\.venv\Scripts\python.exe -X utf8 -m harness.live --probe --adb-address 127.0.0.1:16384
```
## 当前覆盖

- 屏幕识别和普通回合标定覆盖战场 #02、#04、#05、#06、#09、#12。部分领袖回合沿用通用标定，实机验证仍未完成。
- 基本实现完整的一局。