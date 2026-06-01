import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langgraph_persuasion_guard.modeling import RoleModelConfig, build_models


class ModelingTests(unittest.TestCase):
    def test_build_models_from_python_defaults_without_env(self):
        with patch(
            "langgraph_persuasion_guard.modeling.build_model", side_effect=lambda cfg: cfg
        ):
            models = build_models(
                default_model="gpt-4o-mini",
                default_provider="openai",
                use_env=False,
            )

        self.assertEqual(models["router"], RoleModelConfig("gpt-4o-mini", "openai", 0.0, None))
        self.assertEqual(models["sanitizer"], RoleModelConfig("gpt-4o-mini", "openai", 0.2, None))
        self.assertEqual(models["executor"], RoleModelConfig("gpt-4o-mini", "openai", 0.0, None))
        self.assertEqual(models["chat"], RoleModelConfig("gpt-4o-mini", "openai", 0.7, None))

    def test_build_models_from_role_overrides_without_env(self):
        overrides = {
            "router": RoleModelConfig("r", "openai", 0.0, 100),
            "sanitizer": RoleModelConfig("s", "openai", 0.2, 200),
            "executor": RoleModelConfig("e", "openai", 0.0, 300),
            "chat": RoleModelConfig("c", "openai", 0.7, 400),
        }
        with patch(
            "langgraph_persuasion_guard.modeling.build_model", side_effect=lambda cfg: cfg
        ):
            models = build_models(role_overrides=overrides, use_env=False)

        self.assertEqual(models, overrides)

    def test_chat_max_tokens_overrides_chat_role_only(self):
        overrides = {
            "router": RoleModelConfig("r", "openai", 0.0, 100),
            "sanitizer": RoleModelConfig("s", "openai", 0.2, 200),
            "executor": RoleModelConfig("e", "openai", 0.0, 300),
            "chat": RoleModelConfig("c", "openai", 0.7, 400),
        }
        with patch(
            "langgraph_persuasion_guard.modeling.build_model", side_effect=lambda cfg: cfg
        ):
            models = build_models(
                role_overrides=overrides,
                chat_max_tokens=1234,
                use_env=False,
            )

        self.assertEqual(models["chat"], RoleModelConfig("c", "openai", 0.7, 1234))
        self.assertEqual(models["router"], overrides["router"])
        self.assertEqual(models["sanitizer"], overrides["sanitizer"])
        self.assertEqual(models["executor"], overrides["executor"])

    def test_build_models_raises_when_no_source_for_model(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                build_models(use_env=False)
        self.assertIn("Unable to resolve model for role 'router'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
