from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "tools" / "soul-research-console.html"


class ResearchConsoleTests(unittest.TestCase):
    def test_console_is_a_local_read_only_viewer(self):
        source = PAGE.read_text(encoding="utf-8").lower()
        self.assertNotIn("<script src=", source)
        self.assertNotIn("fetch(", source)
        self.assertNotIn("xmlhttprequest", source)
        self.assertIn("read-only viewer", source)
        self.assertIn("no model code executes", source)

    def test_console_escapes_untrusted_export_values_before_html_rendering(self):
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn("function escapeHtml(value)", source)
        for dynamic_value in (
            "escapeHtml(c[0])",
            "escapeHtml(c[1])",
            "escapeHtml(c[2])",
            "escapeHtml(k)",
            "escapeHtml(x[1])",
            "escapeHtml(e.message)",
        ):
            self.assertIn(dynamic_value, source)

    def test_console_loads_both_checkpoint_reports_and_metrics(self):
        source = PAGE.read_text(encoding="utf-8")
        for required in (
            "A/metrics.json", "B1/metrics.json",
            "A/report.json", "B1/report.json",
            "model_aggregate_equal_episode",
            "persistence_aggregate_equal_episode",
            "mean_delta_aggregate_equal_episode",
            "checkpoint_sha256", "pilot_sha256",
        ):
            self.assertIn(required, source)

    def test_research_guardrails_are_visible(self):
        source = PAGE.read_text(encoding="utf-8")
        for required in ("INCONCLUSIVE", "TO_LOCK", "Confirm", "optimizer"):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
