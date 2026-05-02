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

    def test_stop_navigation(self) -> None:
        nav = NavigationStateMachine()
        nav.command("开始过马路")
        result = nav.command("停止过马路")
        self.assertEqual(result.mode, NavigationMode.IDLE)


if __name__ == "__main__":
    unittest.main()
