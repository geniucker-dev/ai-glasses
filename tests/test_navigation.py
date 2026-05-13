import unittest

from aiglasses.navigation import NavigationMode, NavigationStateMachine
from aiglasses.vision.tuning import VisionTuning


class NavigationTests(unittest.TestCase):
    def _green_crossing_observation(self) -> dict[str, object]:
        return {
            "frame_width": 640,
            "frame_height": 480,
            "crosswalk": {
                "center_offset": 0.0,
                "area_ratio": 0.20,
                "contour": [(0.2, 0.55), (0.8, 0.55), (0.8, 0.95), (0.2, 0.95)],
            },
            "traffic_light": "go",
        }

    def _arrival_crosswalk_observation(self) -> dict[str, object]:
        return {
            "frame_width": 640,
            "frame_height": 480,
            "crosswalk": {
                "center_offset": 0.0,
                "area_ratio": 0.04,
                "contour": [(0.3, 0.10), (0.7, 0.10), (0.7, 0.30), (0.3, 0.30)],
            },
            "traffic_light": "go",
        }

    def _distant_crossing_observation(self) -> dict[str, object]:
        return {
            "frame_width": 640,
            "frame_height": 480,
            "crosswalk": {
                "center_offset": 0.0,
                "area_ratio": 0.04,
                "contour": [(0.3, 0.10), (0.7, 0.10), (0.7, 0.30), (0.3, 0.30)],
            },
            "traffic_light": "go",
        }

    def _blind_path_crosswalk_observation(self) -> dict[str, object]:
        return {
            "frame_width": 640,
            "frame_height": 480,
            "blind_path": {"center_offset": 0.0, "angle_deg": 0},
            "crosswalk": {
                "center_offset": 0.0,
                "area_ratio": 0.04,
                "contour": [(0.25, 0.35), (0.75, 0.35), (0.75, 0.55), (0.25, 0.55)],
            },
        }

    def _vehicle_hazard_observation(self) -> dict[str, object]:
        return {
            "frame_width": 640,
            "frame_height": 480,
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

    def _start_active_crossing(self, nav: NavigationStateMachine) -> None:
        nav.command("开始过马路")
        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation(self._green_crossing_observation()).speech,
            "绿灯稳定，开始通行。",
        )

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

    def test_blind_path_crosswalk_warns_approaching_road(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")

        result = nav.process_observation(self._blind_path_crosswalk_observation())

        self.assertEqual(result.speech, "前方到马路了，请先停下。")
        self.assertEqual(result.mode, NavigationMode.BLIND_PATH)

    def test_blind_path_distant_crosswalk_warns_without_stopping(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        observation = self._blind_path_crosswalk_observation()
        observation["crosswalk"] = {
            "center_offset": 0.0,
            "area_ratio": 0.006,
            "contour": [(0.30, 0.20), (0.70, 0.20), (0.70, 0.40), (0.30, 0.40)],
        }

        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "前方发现路口，请注意红绿灯。")
        self.assertEqual(result.mode, NavigationMode.BLIND_PATH)

    def test_blind_path_crosswalk_stop_does_not_downgrade_to_distant_warning(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始导航")

        self.assertEqual(
            nav.process_observation(self._blind_path_crosswalk_observation()).speech,
            "前方到马路了，请先停下。",
        )
        observation = self._blind_path_crosswalk_observation()
        observation["crosswalk"] = {
            "center_offset": 0.0,
            "area_ratio": 0.006,
            "contour": [(0.30, 0.20), (0.70, 0.20), (0.70, 0.40), (0.30, 0.40)],
        }

        result = nav.process_observation(observation)

        self.assertIsNone(result.speech)
        self.assertEqual(result.state["last_speech"], "前方到马路了，请先停下。")

    def test_blind_path_crosswalk_stop_distance_is_tunable(self) -> None:
        nav = NavigationStateMachine(tuning=VisionTuning(road_stop_bottom_min=0.35))
        nav.command("开始导航")
        observation = self._blind_path_crosswalk_observation()
        observation["crosswalk"] = {
            "center_offset": 0.0,
            "area_ratio": 0.020,
            "contour": [(0.30, 0.20), (0.70, 0.20), (0.70, 0.40), (0.30, 0.40)],
        }

        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "前方到马路了，请先停下。")

    def test_blind_path_crosswalk_warning_bypasses_cooldown(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始导航")

        self.assertEqual(
            nav.process_observation({"blind_path": {"center_offset": 0.0, "angle_deg": 0}}).speech,
            "保持直行。",
        )
        result = nav.process_observation(self._blind_path_crosswalk_observation())

        self.assertEqual(result.speech, "前方到马路了，请先停下。")

    def test_blind_path_crosswalk_warning_keeps_obstacle_priority(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        observation = self._blind_path_crosswalk_observation()
        observation["obstacles"] = [
            {
                "label": "car",
                "confidence": 0.9,
                "box": (270, 210, 370, 430),
                "area_ratio": 0.04,
            }
        ]

        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "前方盲道上疑似有车，请先停下。")

    def test_missing_blind_path_with_crosswalk_warns_approaching_road(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始导航")
        observation = self._blind_path_crosswalk_observation()
        observation.pop("blind_path")

        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "前方到马路了，请先停下。")

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
        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation(self._green_crossing_observation()).speech,
            "绿灯稳定，开始通行。",
        )

    def test_crossing_ignores_obstacle_model_outputs_by_default_before_go(self) -> None:
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

        self.assertEqual(result.speech, "发现斑马线，对准方向。")

    def test_green_crossing_ignores_obstacle_model_outputs_after_go_by_default(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation(self._green_crossing_observation()).speech,
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

        self.assertIsNone(result.speech)

    def test_green_crossing_with_side_vehicle_can_go(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        observation = self._green_crossing_observation()
        observation["obstacles"] = [
            {
                "label": "car",
                "confidence": 0.9,
                "box": (560, 160, 635, 360),
                "area_ratio": 0.05,
            }
        ]

        result = nav.process_observation(observation)

        self.assertIsNone(result.speech)
        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "绿灯稳定，开始通行。")

    def test_green_crossing_with_two_wheel_obstacles_waits_when_enabled(self) -> None:
        for label in ("bicycle", "motorcycle", "scooter"):
            with self.subTest(label=label):
                nav = NavigationStateMachine(
                    clock=lambda: 10.0,
                    tuning=VisionTuning(crossing_obstacles_enabled=True),
                )
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

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation(self._green_crossing_observation()).speech,
            "绿灯稳定，开始通行。",
        )

    def test_crossing_red_resets_green_stability(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation({"traffic_light": "stop"}).speech,
            "红灯。",
        )
        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)

    def test_crossing_crosswalk_loss_resets_green_stability(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertIsNone(nav.process_observation({"traffic_light": "go"}).speech)
        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        self.assertEqual(
            nav.process_observation(self._green_crossing_observation()).speech,
            "绿灯稳定，开始通行。",
        )

    def test_crossing_loss_before_green_does_not_complete(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        result = nav.process_observation({})
        for _ in range(5):
            result = nav.process_observation({})

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNone(result.speech)
        state = result.state
        assert state is not None
        self.assertFalse(state["crossing_active"])
        self.assertEqual(state["crossing_completion_candidate_frames"], 0)

    def test_crossing_starts_after_stable_green(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        result = nav.process_observation(self._green_crossing_observation())

        self.assertEqual(result.speech, "绿灯稳定，开始通行。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertTrue(result.state["crossing_active"])

    def test_crossing_dropout_after_active_does_not_complete(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(8):
            result = nav.process_observation({"frame_width": 640, "frame_height": 480})

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNone(result.speech)
        self.assertIsNotNone(result.state)
        self.assertTrue(result.state["crossing_active"])
        self.assertEqual(result.state["crossing_lost_crosswalk_frames"], 8)

    def test_crossing_partial_empty_observations_do_not_complete(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(8):
            result = nav.process_observation({})

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNone(result.speech)

    def test_crossing_ignores_transient_crosswalk_loss(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(3):
            result = nav.process_observation({"frame_width": 640, "frame_height": 480})
            self.assertIsNone(result.speech)
            self.assertEqual(result.mode, NavigationMode.CROSSING)

        result = nav.process_observation({"crosswalk": {"center_offset": 0.0}})

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state["crossing_lost_crosswalk_frames"], 0)

    def test_crossing_obstacle_model_output_is_ignored_by_default_after_crossing_starts(
        self,
    ) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        result = nav.process_observation(self._vehicle_hazard_observation())

        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state["crossing_clear_path_frames"], 1)
        self.assertEqual(result.state["crossing_completion_candidate_frames"], 0)

    def test_crossing_vehicle_obstacle_blocks_completion_when_enabled(self) -> None:
        nav = NavigationStateMachine(
            clock=lambda: 10.0,
            tuning=VisionTuning(crossing_obstacles_enabled=True),
        )
        self._start_active_crossing(nav)

        result = nav.process_observation(self._vehicle_hazard_observation())

        self.assertEqual(
            result.speech,
            "绿灯，但斑马线附近疑似有车，请先等待，确认安全后再过街。",
        )
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        for _ in range(3):
            self.assertIsNone(nav.process_observation(self._vehicle_hazard_observation()).speech)

        for _ in range(8):
            result = nav.process_observation({"frame_width": 640, "frame_height": 480})

        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)

    def test_crossing_non_vehicle_obstacle_waits_when_enabled(self) -> None:
        nav = NavigationStateMachine(
            clock=lambda: 10.0,
            tuning=VisionTuning(crossing_obstacles_enabled=True),
        )
        nav.command("开始过马路")

        result = nav.process_observation(
            {
                "frame_width": 640,
                "frame_height": 480,
                "crosswalk": {"center_offset": 0.0},
                "traffic_light": "go",
                "obstacles": [
                    {
                        "label": "bench",
                        "confidence": 0.9,
                        "box": (250, 160, 390, 360),
                        "area_ratio": 0.06,
                    }
                ],
            }
        )

        self.assertEqual(
            result.speech,
            "绿灯，但斑马线附近疑似有长椅，请先等待，确认安全后再过街。",
        )

    def test_crossing_arrival_evidence_warns_without_exiting(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(3):
            self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        result = nav.process_observation(self._arrival_crosswalk_observation())
        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        result = nav.process_observation(self._arrival_crosswalk_observation())
        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        result = nav.process_observation(self._arrival_crosswalk_observation())

        self.assertEqual(result.speech, "疑似已通过人行横道，请确认安全后停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)

    def test_crossing_static_distant_crosswalk_does_not_start(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        nav.command("开始过马路")

        for _ in range(8):
            result = nav.process_observation(self._distant_crossing_observation())
            self.assertNotEqual(result.speech, "绿灯稳定，开始通行。")

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertFalse(result.state["crossing_active"])
        self.assertEqual(result.state["crossing_green_frames"], 0)
        self.assertFalse(result.state["crossing_saw_near_crosswalk"])

    def test_crossing_timeout_prevents_late_completion(self) -> None:
        now = 10.0

        def clock() -> float:
            return now

        nav = NavigationStateMachine(clock=clock)
        self._start_active_crossing(nav)
        now = 56.0

        result = nav.process_observation(self._arrival_crosswalk_observation())

        self.assertEqual(result.speech, "过马路时间较长，请确认周围安全，必要时停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)

    def test_crossing_does_not_complete_without_arrival_area_ratio(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(3):
            self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        observation = self._arrival_crosswalk_observation()
        crosswalk = observation["crosswalk"]
        self.assertIsInstance(crosswalk, dict)
        crosswalk.pop("area_ratio")
        for _ in range(3):
            result = nav.process_observation(observation)

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNone(result.speech)

    def test_crossing_arrival_evidence_after_light_leaves_view_warns_without_exiting(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        for _ in range(3):
            self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        observation = self._arrival_crosswalk_observation()
        observation.pop("traffic_light")
        result = nav.process_observation(observation)
        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        result = nav.process_observation(observation)
        self.assertIsNone(result.speech)
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "疑似已通过人行横道，请确认安全后停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)

    def test_crossing_obstacle_without_green_uses_neutral_wait_message(self) -> None:
        nav = NavigationStateMachine(
            clock=lambda: 10.0,
            tuning=VisionTuning(crossing_obstacles_enabled=True),
        )
        nav.command("开始过马路")

        observation = self._vehicle_hazard_observation()
        observation.pop("traffic_light")
        result = nav.process_observation(observation)

        self.assertEqual(result.speech, "斑马线附近疑似有车，请先等待。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)

    def test_crossing_red_light_priority_over_completion(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        result = nav.process_observation({"traffic_light": "stop"})

        self.assertEqual(result.speech, "红灯。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state["crossing_lost_crosswalk_frames"], 0)
        self.assertEqual(result.state["crossing_clear_path_frames"], 0)
        for _ in range(3):
            self.assertIsNone(nav.process_observation({"traffic_light": "stop"}).speech)

    def test_crossing_transient_hazard_is_ignored_by_default_after_crossing_starts(self) -> None:
        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)

        self.assertIsNone(nav.process_observation(self._green_crossing_observation()).speech)
        result = nav.process_observation(self._vehicle_hazard_observation())
        self.assertIsNone(result.speech)
        result = nav.process_observation(self._arrival_crosswalk_observation())
        self.assertIsNone(result.speech)
        result = nav.process_observation(self._arrival_crosswalk_observation())
        self.assertIsNone(result.speech)
        result = nav.process_observation(self._arrival_crosswalk_observation())
        self.assertIsNone(result.speech)
        result = nav.process_observation(self._arrival_crosswalk_observation())

        self.assertEqual(result.speech, "疑似已通过人行横道，请确认安全后停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)

        nav = NavigationStateMachine(clock=lambda: 10.0)
        self._start_active_crossing(nav)
        nav.process_observation({"frame_width": 640, "frame_height": 480})
        nav.process_observation({"frame_width": 640, "frame_height": 480})

        nav.command("停止过马路")
        nav.command("开始过马路")
        result = nav.process_observation({"frame_width": 640, "frame_height": 480})

        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertFalse(result.state["crossing_active"])
        self.assertEqual(result.state["crossing_lost_crosswalk_frames"], 0)

    def test_crossing_active_timeout_warns_without_exiting(self) -> None:
        now = 10.0

        def clock() -> float:
            return now

        nav = NavigationStateMachine(clock=clock)
        self._start_active_crossing(nav)
        now = 56.0

        result = nav.process_observation({"crosswalk": {"center_offset": 0.0}})

        self.assertEqual(result.speech, "过马路时间较长，请确认周围安全，必要时停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)
        self.assertIsNotNone(result.state)
        self.assertTrue(result.state["crossing_active"])

    def test_crossing_active_timeout_does_not_exit_on_empty_observations(self) -> None:
        now = 10.0

        def clock() -> float:
            return now

        nav = NavigationStateMachine(clock=clock)
        self._start_active_crossing(nav)
        now = 56.0

        result = nav.process_observation({})

        self.assertEqual(result.speech, "过马路时间较长，请确认周围安全，必要时停止过马路模式。")
        self.assertEqual(result.mode, NavigationMode.CROSSING)
        self.assertEqual(nav.mode, NavigationMode.CROSSING)

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
