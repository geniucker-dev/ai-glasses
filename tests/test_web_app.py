import unittest


class WebAppTests(unittest.TestCase):
    def test_web_app_imports(self) -> None:
        from aiglasses.web.app import create_app

        self.assertTrue(callable(create_app))


if __name__ == "__main__":
    unittest.main()
