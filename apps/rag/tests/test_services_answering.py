from __future__ import annotations

import types
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from apps.rag.services.answering import AnswerOutput, synthesize_answer


class AnsweringServiceTests(SimpleTestCase):
    @override_settings(RAG_ANSWER_BACKEND="fallback")
    def test_fallback_returns_contract_for_empty_context(self) -> None:
        result = synthesize_answer(
            query_text="where is auth handled?",
            contexts=[],
            citations=[],
        )
        self.assertIsInstance(result, AnswerOutput)
        self.assertEqual(result.contract_version, "v1")
        self.assertEqual(result.backend, "fallback")
        self.assertEqual(result.status, "fallback")
        self.assertIn("could not find enough relevant context", result.answer_text.lower())

    @override_settings(RAG_ANSWER_BACKEND="fallback")
    def test_fallback_returns_preview_and_doc_numbers(self) -> None:
        result = synthesize_answer(
            query_text="where is auth handled?",
            contexts=["def login_view(request):\n    ..."],
            citations=[{"file_path": "apps/accounts/views.py", "start_line": 10, "end_line": 30}],
        )
        self.assertEqual(result.backend, "fallback")
        self.assertIn("Top retrieved context preview", result.answer_text)
        self.assertEqual(result.citations_doc_numbers, [1])

    @override_settings(RAG_ANSWER_BACKEND="unknown_backend")
    def test_unknown_backend_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            synthesize_answer("q", ["ctx"], [])

    @override_settings(RAG_ANSWER_BACKEND="openai", OPENAI_API_KEY="test-key")
    def test_openai_json_output_is_parsed_into_contract(self) -> None:
        class _FakeResponse:
            output_text = (
                '{"answer_text":"Auth is handled in apps/accounts/views.py [DOC 1].",'
                '"key_points":["Uses django.contrib.auth.authenticate.","Calls login() after validation."],'
                '"citations_doc_numbers":[1],'
                '"insufficient_context":false}'
            )

        class _FakeResponses:
            @staticmethod
            def create(**kwargs):  # noqa: ARG004
                return _FakeResponse()

        class _FakeOpenAI:
            def __init__(self, api_key=None):  # noqa: ARG002
                self.responses = _FakeResponses()

        fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            result = synthesize_answer(
                query_text="where is auth handled?",
                contexts=["ctx1"],
                citations=[{"file_path": "apps/accounts/views.py", "start_line": 1, "end_line": 10}],
            )

        self.assertEqual(result.backend, "openai")
        self.assertEqual(result.status, "ok")
        self.assertIn("Auth is handled", result.answer_text)
        self.assertEqual(result.citations_doc_numbers, [1])
        self.assertGreaterEqual(len(result.key_points), 1)

    @override_settings(RAG_ANSWER_BACKEND="openai", OPENAI_API_KEY="test-key")
    def test_openai_non_json_output_falls_back_to_text_contract(self) -> None:
        class _FakeResponse:
            output_text = "Auth appears in auth_manager.py [DOC 2]"

        class _FakeResponses:
            @staticmethod
            def create(**kwargs):  # noqa: ARG004
                return _FakeResponse()

        class _FakeOpenAI:
            def __init__(self, api_key=None):  # noqa: ARG002
                self.responses = _FakeResponses()

        fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)

        with patch.dict("sys.modules", {"openai": fake_openai_module}):
            result = synthesize_answer(
                query_text="where is auth handled?",
                contexts=["ctx1"],
                citations=[{"file_path": "a.py", "start_line": 1, "end_line": 2}],
            )

        self.assertEqual(result.backend, "openai")
        self.assertEqual(result.status, "ok")
        self.assertIn("auth_manager.py", result.answer_text)
        self.assertEqual(result.citations_doc_numbers, [2])
