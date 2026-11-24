"""Replay partial leader previews against the local engine without game input."""

from harness.sidecar import Sidecar, SidecarError


def main():
    state = {"config": {"modeId": "mode_single_13"}}
    sidecar = Sidecar()
    try:
        for round_number, boss, leader, fixed_escort in (
            (14, "boss_1", "enemy_9013_acstmk", "enemy_1006_shield_2"),
            (15, "boss_8", "enemy_9013_acstmk_2", "enemy_10027_vtsk"),
        ):
            args = {"round": round_number, "bossId": boss}
            observed = ["enemy_1043_zomsbr", leader]
            full = [*observed, fixed_escort]
            recovered = sidecar.request("preview", state, kinds=observed, **args)
            exact = sidecar.request("preview", state, kinds=full, **args)
            assert recovered["preview"] == exact["preview"]
            assert {row["enemyKey"] for row in recovered["preview"]} == set(full)

            # The invariant escort cannot identify the variable enemy pick.
            try:
                sidecar.request("preview", state, kinds=[leader], **args)
            except SidecarError as error:
                assert "no template pick matches" in str(error)
            else:
                raise AssertionError("Accepted a leader preview with no variable enemy observed")
    finally:
        sidecar.close()
    print("Leader preview replay passed")


if __name__ == "__main__":
    main()
