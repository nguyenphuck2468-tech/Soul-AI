"""Guard development code against reading the held-out shock catalogue."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "shocks_test",
    "test_new_lava",
    "test_inverted_move",
    "test_tree_dry",
)
SCAN_ROOTS = ("scripts", "skills", "monitor", "safety")


class NoTestShockLeakTests(unittest.TestCase):
    def test_dev_code_does_not_reference_held_out_shocks(self):
        candidates: list[Path] = []
        for name in SCAN_ROOTS:
            root = ROOT / name
            if root.exists():
                candidates.extend(path for path in root.rglob("*") if path.is_file())

        bench = ROOT / "bench"
        if bench.exists():
            for root in bench.glob("dev_*"):
                candidates.extend(
                    path for path in (root.rglob("*") if root.is_dir() else (root,))
                    if path.is_file()
                )

        leaked = []
        for path in candidates:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for token in FORBIDDEN:
                if token in content:
                    leaked.append(f"{path.relative_to(ROOT)} contains {token}")
        self.assertEqual(leaked, [], "held-out shock references found:\n" + "\n".join(leaked))


if __name__ == "__main__":
    unittest.main()
