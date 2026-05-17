from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
COMPILE_PATH = HERE / "compile.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compile_mod = _load_module(COMPILE_PATH, "rlm_trace_compile_under_test")


class CompileTraceTest(unittest.TestCase):
    def _write_trace(self, tmpdir: Path, *records: dict) -> Path:
        path = tmpdir / "trace.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return path

    def _compile_trace(self, tmpdir: Path, *records: dict, **kwargs: object) -> Path:
        trace_path = self._write_trace(tmpdir, *records)
        out_path = tmpdir / "artifact.py"
        compile_mod.compile_trace(log_path=trace_path, out_path=out_path, **kwargs)
        return out_path

    def test_live_llm_queries_inherit_run_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            out_path = self._compile_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "FINAL(llm_query('prompt without per-call model'))",
                            "result": {"stdout": "", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": None,
                },
                emit_readable=False,
            )
            artifact = _load_module(out_path, "compiled_live_model_test")

            seen_models: list[str | None] = []

            def fake_chat_completion(prompt: str, model: str | None = None) -> str:
                seen_models.append(model)
                return f"model={model}"

            artifact._openai_chat_completion = fake_chat_completion

            result = artifact.run(
                "",
                llm_mode="live",
                model="requested-model",
                verbose=False,
                echo_code_output=False,
            )

            self.assertEqual(result, "model=requested-model")
            self.assertEqual(seen_models, ["requested-model"])

    def test_successful_stderr_output_is_not_an_expected_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            out_path = self._compile_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "import sys\nprint('warning only', file=sys.stderr)\nprint('ok')",
                            "result": {
                                "stdout": "ok\n",
                                "stderr": "warning only\n",
                                "rlm_calls": [],
                            },
                        }
                    ],
                    "final_answer": "ok",
                },
                emit_readable=False,
            )
            artifact = _load_module(out_path, "compiled_stderr_success_test")

            with contextlib.redirect_stderr(io.StringIO()):
                result = artifact.run("", verbose=False, echo_code_output=False)

            self.assertEqual(result, "ok")
            self.assertEqual(artifact.TRACE_BLOCKS[0][2], None)

    def test_stderr_exception_summary_remains_expected_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            out_path = self._compile_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "raise ValueError('boom')",
                            "result": {
                                "stdout": "",
                                "stderr": "\nValueError: boom",
                                "rlm_calls": [],
                            },
                        },
                        {
                            "code": "print('recovered')",
                            "result": {
                                "stdout": "recovered\n",
                                "stderr": "",
                                "rlm_calls": [],
                            },
                        },
                    ],
                    "final_answer": "recovered",
                },
                emit_readable=False,
            )
            artifact = _load_module(out_path, "compiled_expected_failure_test")

            result = artifact.run("", verbose=False, echo_code_output=False)

            self.assertEqual(result, "recovered")
            self.assertEqual(artifact.TRACE_BLOCKS[0][2], "ValueError: boom")

    def test_emit_readable_false_suppresses_explicit_readable_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            readable_path = tmpdir / "explicit_recovered.py"
            self._compile_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "print('ok')",
                            "result": {"stdout": "ok\n", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": "ok",
                },
                readable_out_path=readable_path,
                emit_readable=False,
            )

            self.assertFalse(readable_path.exists())

    def test_no_readable_out_suppresses_cli_readable_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            trace_path = self._write_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "print('ok')",
                            "result": {"stdout": "ok\n", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": "ok",
                },
            )
            out_path = tmpdir / "artifact.py"
            readable_path = tmpdir / "explicit_recovered.py"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = compile_mod.main(
                    [
                        str(trace_path),
                        str(out_path),
                        "--readable-out",
                        str(readable_path),
                        "--no-readable-out",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_path.exists())
            self.assertFalse(readable_path.exists())

    def test_live_mode_does_not_require_trace_final_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            out_path = self._compile_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "llm_text = llm_query('new input prompt')\nprint(llm_text)",
                            "result": {
                                "stdout": "recorded\n",
                                "stderr": "",
                                "rlm_calls": [
                                    {
                                        "prompt": "new input prompt",
                                        "response": "recorded",
                                    }
                                ],
                            },
                        }
                    ],
                    "final_answer": "recorded",
                },
                emit_readable=False,
            )
            artifact = _load_module(out_path, "compiled_live_trace_final_relaxed_test")
            artifact._openai_chat_completion = lambda prompt, model=None: "live-response"

            result = artifact.run(
                "",
                llm_mode="live",
                verbose=False,
                echo_code_output=False,
            )
            self.assertEqual(result, "live-response")
            with self.assertRaises(AssertionError):
                artifact.verify_trace_final(
                    "",
                    llm_mode="live",
                    verbose=False,
                    echo_code_output=False,
                )


if __name__ == "__main__":
    unittest.main()
