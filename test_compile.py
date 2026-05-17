from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from unittest import mock
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

    def _compile_minimal_artifact(self, tmpdir: Path) -> ModuleType:
        out_path = self._compile_trace(
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
            emit_readable=False,
        )
        return _load_module(out_path, f"compiled_minimal_{id(tmpdir)}")

    def test_live_openai_api_key_uses_openai_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            artifact = self._compile_minimal_artifact(tmpdir)
            captured: dict[str, object] = {}

            class FakeResponse:
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

                def read(self) -> bytes:
                    return b'{"choices":[{"message":{"content":"ok"}}]}'

            def fake_urlopen(request: object, timeout: int) -> FakeResponse:
                captured["url"] = request.full_url
                captured["authorization"] = request.get_header("Authorization")
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                captured["timeout"] = timeout
                return FakeResponse()

            with mock.patch.dict(artifact.os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=True):
                with mock.patch.object(artifact.urllib.request, "urlopen", fake_urlopen):
                    self.assertEqual(artifact._chat_completion("prompt"), "ok")

            self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
            self.assertEqual(captured["authorization"], "Bearer sk-openai")
            self.assertEqual(captured["payload"]["model"], "gpt-5-mini")

    def test_live_openrouter_api_key_uses_openrouter_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            artifact = self._compile_minimal_artifact(tmpdir)
            captured: dict[str, object] = {}

            class FakeResponse:
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

                def read(self) -> bytes:
                    return b'{"choices":[{"message":{"content":"ok"}}]}'

            def fake_urlopen(request: object, timeout: int) -> FakeResponse:
                captured["url"] = request.full_url
                captured["authorization"] = request.get_header("Authorization")
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                return FakeResponse()

            with mock.patch.dict(artifact.os.environ, {"OPENROUTER_API_KEY": "sk-or"}, clear=True):
                with mock.patch.object(artifact.urllib.request, "urlopen", fake_urlopen):
                    self.assertEqual(artifact._chat_completion("prompt"), "ok")

            self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(captured["authorization"], "Bearer sk-or")
            self.assertEqual(captured["payload"]["model"], "openai/gpt-5-mini")

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

            artifact._chat_completion = fake_chat_completion

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
            artifact._chat_completion = lambda prompt, model=None: "live-response"

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


    def test_recovered_file_is_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            trace_path = self._write_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "result = context.upper()\nprint(result)",
                            "result": {"stdout": "HELLO\n", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": "HELLO",
                },
            )
            out_path = tmpdir / "artifact.py"
            compile_mod.compile_trace(log_path=trace_path, out_path=out_path)
            recovered_path = tmpdir / "artifact_recovered.py"
            self.assertTrue(recovered_path.exists())
            recovered = _load_module(recovered_path, "recovered_runnable_test")
            self.assertTrue(hasattr(recovered, "run"))
            result = recovered.run("hello")
            self.assertEqual(result, "HELLO")

    def test_recovered_file_final_var_resolves_exec_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            trace_path = self._write_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": 'result = context.upper()\nFINAL_VAR("result")',
                            "result": {"stdout": "", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": "HELLO",
                },
            )
            out_path = tmpdir / "artifact.py"
            compile_mod.compile_trace(log_path=trace_path, out_path=out_path)
            recovered_path = tmpdir / "artifact_recovered.py"
            recovered = _load_module(recovered_path, "recovered_final_var_test")

            result = recovered.run("hello")

            self.assertEqual(result, "HELLO")

    def test_recovered_file_allows_both_triple_quote_delimiters(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            code = "\n".join(
                [
                    'single = ' + "'''single triple quoted text'''",
                    'double = ' + '"""double triple quoted text"""',
                    'print("ok")',
                ]
            )
            trace_path = self._write_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": code,
                            "result": {"stdout": "ok\n", "stderr": "", "rlm_calls": []},
                        }
                    ],
                    "final_answer": "ok",
                },
            )
            out_path = tmpdir / "artifact.py"
            compile_mod.compile_trace(log_path=trace_path, out_path=out_path)
            recovered_path = tmpdir / "artifact_recovered.py"
            recovered = _load_module(recovered_path, "recovered_triple_quote_test")

            result = recovered.run("")

            self.assertEqual(result, "ok")

    def test_recovered_file_with_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            trace_path = self._write_trace(
                tmpdir,
                {
                    "type": "iteration",
                    "iteration": 1,
                    "code_blocks": [
                        {
                            "code": "answer_text = llm_query('classify: ' + context)\nprint(answer_text)",
                            "result": {
                                "stdout": "positive\n",
                                "stderr": "",
                                "rlm_calls": [
                                    {
                                        "prompt": "classify: test input",
                                        "response": "positive",
                                    }
                                ],
                            },
                        }
                    ],
                    "final_answer": "positive",
                },
            )
            out_path = tmpdir / "artifact.py"
            compile_mod.compile_trace(log_path=trace_path, out_path=out_path)
            recovered_path = tmpdir / "artifact_recovered.py"
            recovered = _load_module(recovered_path, "recovered_llm_test")
            recovered._chat_completion = lambda prompt, model=None: "mocked"
            result = recovered.run("test input")
            self.assertEqual(result, "mocked")


if __name__ == "__main__":
    unittest.main()
