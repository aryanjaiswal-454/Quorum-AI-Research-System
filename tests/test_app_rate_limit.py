from pathlib import Path
import time
import unittest
from unittest.mock import patch

import httpx
from groq import RateLimitError
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class RateLimitUiTests(unittest.TestCase):
    def test_rate_limit_shows_clean_sixty_second_countdown(self) -> None:
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com"),
        )
        error = RateLimitError(
            "raw provider error details",
            response=response,
            body={},
        )
        app = AppTest.from_file(APP_PATH)
        app.session_state["running"] = True
        app.session_state["topic_input"] = "test topic"

        with patch("pipeline.run_research_pipeline", side_effect=error):
            app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("temporarily rate-limited" in item.value for item in app.info)
        )
        self.assertTrue(
            any("seconds" in item.value for item in app.info)
        )
        self.assertFalse(app.error)
        self.assertNotIn("raw provider error details", str(app.info))
        self.assertIsNotNone(app.session_state["rate_limit_retry_at"])

    def test_retry_button_appears_after_sixty_second_wait(self) -> None:
        app = AppTest.from_file(APP_PATH)
        app.session_state["rate_limit_retry_at"] = time.time() - 1

        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("wait is over" in item.value for item in app.success)
        )
        self.assertTrue(any(item.label == "Try again" for item in app.button))


if __name__ == "__main__":
    unittest.main()
