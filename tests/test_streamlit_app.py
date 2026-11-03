"""Exercise navigation, form submission, persistence, and original click flow."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def app(self):
        # First import of SciPy on a fresh environment can take over 20 seconds.
        app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_gaussian_run_reproduce_validate_and_persist(self):
        app = self.app().switch_page("pages/Gaussian_Bandit.py").run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[1].value, "Alternative 3")
        first = app.session_state["gaussian_result"].history.copy()
        app.button[0].click().run()
        self.assertTrue(first.equals(app.session_state["gaussian_result"].history))
        app.number_input[3].set_value(0.0)
        app.button[0].click().run()
        self.assertIn("positive", app.error[0].value)
        self.assertEqual(len(app.exception), 0)
        app.switch_page("pages/Analytics.py").run()
        app.switch_page("pages/Gaussian_Bandit.py").run()
        self.assertEqual(app.metric[1].value, "Alternative 3")
        self.assertEqual(len(app.exception), 0)

    def test_original_click_reset_and_delete(self):
        app = self.app()
        app.button[0].click().run()
        self.assertEqual(len(app.session_state["carousels"]), 1)
        app.switch_page("pages/Interaction.py").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        carousel = next(iter(app.session_state["carousels"].values()))
        self.assertEqual(len(carousel["history"]), 1)
        self.assertEqual(carousel["history"][0]["reward"], 1)
        self.assertEqual(int(carousel["alternatives"].clicks.sum()), 1)
        self.assertEqual(int(carousel["alternatives"].impressions.sum()), carousel["visible_count"])
        next(b for b in app.button if b.label == "No click / Next round").click().run()
        carousel = next(iter(app.session_state["carousels"].values()))
        self.assertEqual(carousel["history"][-1]["reward"], 0)
        self.assertEqual(int(carousel["alternatives"].clicks.sum()), 1)
        self.assertEqual(int(carousel["alternatives"].impressions.sum()), 2 * carousel["visible_count"])
        app.switch_page("pages/Analytics.py").run()
        self.assertEqual(len(app.exception), 0)
        app.switch_page("pages/Create_Carousels.py").run()
        next(b for b in app.button if b.label == "Reset selected carousel").click().run()
        carousel = next(iter(app.session_state["carousels"].values()))
        self.assertEqual(len(carousel["history"]), 0)
        next(b for b in app.button if b.label == "Delete selected carousel").click().run()
        self.assertEqual(len(app.session_state["carousels"]), 0)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
