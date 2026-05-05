import unittest

from aiglasses.navigation import NavigationMode, NavigationStateMachine


class NavigationTests(unittest.TestCase):
    def test_command_starts_blind_path(self) -> None:
        nav = NavigationStateMachine()
        result = nav.command("开始导航")
        self.assertEqual(result.mode, NavigationMode.BLIND_PATH)
        self.assertEqual(result.speech, "盲道导航已启动。")

    def test_blind_path_side_car_does_not_stop(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.0, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (500, 210, 630, 430),
                        "area_ratio": 0.08,
                    }
                ],
            }
        )
        self.assertEqual(result.speech, "保持直行。")

    def test_guidance_for_obstacle_on_blind_path(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.0, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (270, 210, 370, 430),
                        "area_ratio": 0.04,
                    }
                ],
            }
        )
        self.assertEqual(result.speech, "前方盲道上疑似有车，请先停下。")

    def test_guidance_translates_english_obstacle_label(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.0, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "utility pole",
                        "confidence": 0.9,
                        "box": (280, 230, 360, 460),
                        "area_ratio": 0.03,
                    }
                ],
            }
        )
        self.assertEqual(result.speech, "前方盲道上疑似有电线杆，请先停下。")

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

    def test_missing_blind_path_with_centered_near_obstacle_stops_immediately(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        observation = {
            "frame_width": 640,
            "frame_height": 480,
            "obstacles": [
                {
                    "label": "car",
                    "confidence": 0.9,
                    "box": (270, 210, 370, 430),
                    "area_ratio": 0.04,
                }
            ],
        }

        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "前方疑似有车，请先停下。")

    def test_missing_blind_path_ignores_side_obstacle_until_loss_is_stable(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        observation = {
            "frame_width": 640,
            "frame_height": 480,
            "obstacles": [
                {
                    "label": "car",
                    "confidence": 0.9,
                    "box": (500, 210, 630, 430),
                    "area_ratio": 0.08,
                }
            ],
        }

        self.assertIsNone(nav.process_observation(observation).speech)
        self.assertIsNone(nav.process_observation(observation).speech)
        result = nav.process_observation(observation)

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
            nav.process_observation(
                {
                    "frame_width": 640,
                    "frame_height": 480,
                    "blind_path": {"center_offset": 0, "angle_deg": 0},
                    "obstacles": [
                        {
                            "label": "car",
                            "confidence": 0.9,
                            "box": (270, 210, 370, 430),
                            "area_ratio": 0.04,
                        }
                    ],
                }
            ).speech,
            "前方盲道上疑似有车，请先停下。",
        )

    def test_blind_path_obstacle_follows_shifted_path_center(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.45, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "bicycle",
                        "confidence": 0.85,
                        "box": (435, 220, 525, 430),
                        "area_ratio": 0.03,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "前方盲道上疑似有自行车，请先停下。")

    def test_blind_path_obstacle_away_from_shifted_path_does_not_stop(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.45, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (20, 220, 160, 430),
                        "area_ratio": 0.08,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "请向右微调，对准盲道。")

    def test_blind_path_far_tiny_obstacle_does_not_stop(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.0, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "person",
                        "confidence": 0.8,
                        "box": (300, 50, 340, 130),
                        "area_ratio": 0.001,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "保持直行。")

    def test_blind_path_unknown_centered_obstacle_stops(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "blind_path": {"center_offset": 0.0, "angle_deg": 0},
                "obstacles": [
                    {
                        "label": "unknown_object",
                        "confidence": 0.8,
                        "box": (260, 260, 380, 450),
                        "area_ratio": 0.03,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "前方盲道上疑似有unknown_object，请先停下。")

    def test_green_crossing_guidance_bypasses_cooldown(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertEqual(
            nav.process_observation({"crosswalk": {"center_offset": 0.0}}).speech,
            "发现斑马线，对准方向。",
        )
        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )
        self.assertEqual(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech,
            "绿灯稳定，开始通行。",
        )

    def test_green_crossing_with_vehicle_hazard_waits(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "go",
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (230, 160, 410, 360),
                        "area_ratio": 0.08,
                    }
                ],
            }
        )

        self.assertEqual(
            result.speech,
            "绿灯，但斑马线附近疑似有车辆通过，请先等待，确认安全后再过街。",
        )

    def test_green_crossing_with_vehicle_hazard_bypasses_cooldown_after_go(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(
            nav.process_observation(
                {"frame_width": 640, "frame_height": 480, "crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )
        self.assertEqual(
            nav.process_observation(
                {"frame_width": 640, "frame_height": 480, "crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech,
            "绿灯稳定，开始通行。",
        )
        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "go",
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (230, 160, 410, 360),
                        "area_ratio": 0.08,
                    }
                ],
            }
        )

        self.assertEqual(
            result.speech,
            "绿灯，但斑马线附近疑似有车辆通过，请先等待，确认安全后再过街。",
        )

    def test_green_crossing_with_side_vehicle_can_go(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "go",
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (560, 160, 635, 360),
                        "area_ratio": 0.05,
                    }
                ],
            }
        )

        self.assertIsNone(result.speech)
        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "go",
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (560, 160, 635, 360),
                        "area_ratio": 0.05,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "绿灯稳定，开始通行。")

    def test_green_crossing_with_two_wheel_hazards_waits(self) -> None:
        for label in ("bicycle", "motorcycle", "scooter"):
            with self.subTest(label=label):
                nav = NavigationStateMachine(clock=lambda: 10.0)
                nav.command("开始过马路")

                result = nav.process_observation(
                    {
                        "frame_width": 640,
                        "frame_height": 480,
                        "crosswalk": {"center_offset": 0.0},
                        "traffic_light": "go",
                        "obstacles": [
                            {
                                "label": label,
                                "confidence": 0.9,
                                "box": (280, 160, 360, 360),
                                "area_ratio": 0.03,
                            }
                        ],
                    }
                )

                self.assertIn("请先等待", result.speech or "")

    def test_crossing_red_light_keeps_priority_over_vehicle_hazard(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "stop",
                "obstacles": [
                    {
                        "label": "car",
                        "confidence": 0.9,
                        "box": (230, 160, 410, 360),
                        "area_ratio": 0.08,
                    }
                ],
            }
        )

        self.assertEqual(result.speech, "红灯。")

    def test_crossing_red_light_keeps_priority_when_crosswalk_missing(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        result = nav.process_observation({"traffic_light": "stop"})

        self.assertEqual(result.speech, "红灯。")

    def test_crossing_countdown_light_waits(self) -> None:
        for light in ("countdown_go", "countdown_stop"):
            with self.subTest(light=light):
                nav = NavigationStateMachine(clock=lambda: 10.0)
                nav.command("开始过马路")

                result = nav.process_observation(
                    {"crosswalk": {"center_offset": 0.0}, "traffic_light": light}
                )

                self.assertEqual(result.speech, "黄灯。")

    def test_crossing_green_requires_stable_frames(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )
        self.assertEqual(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech,
            "绿灯稳定，开始通行。",
        )

    def test_crossing_red_resets_green_stability(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )
        self.assertEqual(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "stop"}
            ).speech,
            "红灯。",
        )
        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )

    def test_crossing_crosswalk_loss_resets_green_stability(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
        )
        self.assertIsNone(nav.process_observation({"traffic_light": "go"}).speech)
        self.assertIsNone(
            nav.process_observation(
                {"crosswalk": {"center_offset": 0.0}, "traffic_light": "go"}
            ).speech
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
