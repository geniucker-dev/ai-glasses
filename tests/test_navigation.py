import unittest

from aiglasses.navigation import NavigationMode, NavigationStateMachine


class NavigationTests(unittest.TestCase):
    def test_command_starts_blind_path(self) -> None:
        nav = NavigationStateMachine()
        result = nav.command("开始导航")
        self.assertEqual(result.mode, NavigationMode.BLIND_PATH)
        self.assertEqual(result.speech, "盲道导航已启动。")

    def test_guidance_for_obstacle(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        result = nav.process_observation({"nearest_obstacle": {"label": "车"}})
        self.assertEqual(result.speech, "前方有车，停一下。")

    def test_guidance_translates_english_obstacle_label(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        result = nav.process_observation({"nearest_obstacle": {"label": "utility pole"}})
        self.assertEqual(result.speech, "前方有电线杆，停一下。")

    def test_stop_navigation(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始过马路")
        result = nav.command("停止过马路")
        self.assertEqual(result.mode, NavigationMode.IDLE)
        self.assertEqual(result.speech, "过马路模式已停止。")

    def test_generic_stop_uses_current_navigation_mode(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.command("停止检测")

        self.assertEqual(result.mode, NavigationMode.IDLE)
        self.assertEqual(result.speech, "盲道导航已停止。")

    def test_generic_stop_uses_current_traffic_light_mode(self) -> None:
        nav = NavigationStateMachine()
        nav.command("检测红绿灯")

        result = nav.command("停止检测")

        self.assertEqual(result.mode, NavigationMode.IDLE)
        self.assertEqual(result.speech, "红绿灯检测已停止。")

    def test_detection_loss_guidance_requires_stable_frames(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        self.assertIsNone(nav.process_observation({}).speech)
        self.assertIsNone(nav.process_observation({}).speech)
        result = nav.process_observation({})

        self.assertEqual(result.speech, "没看到盲道，请原地小幅转动。")

    def test_guidance_cooldown_suppresses_nonurgent_changes(self) -> None:
        now = 10.0

        def clock() -> float:
            return now

        nav = NavigationStateMachine(clock=clock)
        nav.command("开始导航")
        left = {"blind_path": {"center_offset": -0.3, "angle_deg": 0}}
        right = {"blind_path": {"center_offset": 0.3, "angle_deg": 0}}

        self.assertEqual(nav.process_observation(left).speech, "请向左微调，对准盲道。")
        self.assertIsNone(nav.process_observation(right).speech)

        now = 12.1
        self.assertEqual(nav.process_observation(right).speech, "请向右微调，对准盲道。")

    def test_urgent_obstacle_guidance_bypasses_cooldown(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始导航")

        self.assertEqual(
            nav.process_observation({"blind_path": {"center_offset": 0, "angle_deg": 0}}).speech,
            "保持直行。",
        )
        self.assertEqual(
            nav.process_observation({"nearest_obstacle": {"label": "车"}}).speech,
            "前方有车，停一下。",
        )

    def test_green_crossing_guidance_bypasses_cooldown(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertEqual(
            nav.process_observation({"crosswalk": {"center_offset": 0.0}}).speech,
            "发现斑马线，对准方向。",
        )
        self.assertEqual(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech,
            "绿灯稳定，开始通行。",
        )

    def test_standalone_traffic_light_changes_bypass_cooldown(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("检测红绿灯")

        self.assertEqual(nav.process_observation({"traffic_light": "stop"}).speech, "红灯。")
        self.assertEqual(nav.process_observation({"traffic_light": "go"}).speech, "绿灯。")
        self.assertEqual(
            nav.process_observation({"traffic_light": "countdown_stop"}).speech,
            "黄灯。",
        )


if __name__ == "__main__":
    unittest.main()
