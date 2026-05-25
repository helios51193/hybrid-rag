from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from apps.rag.models import Conversation, ConversationMessage, IndexingJob


class QueryGraphHistoryViewTests(TestCase):
    def setUp(self) -> None:
        self.project_id = "proj-history"
        self.job = IndexingJob.objects.create(
            project_id=self.project_id,
            source_dir="/tmp/repo",
            status=IndexingJob.Status.DONE,
            metadata={"name": "Repo", "source_type": "folder"},
        )
        self.conversation = Conversation.objects.create(
            project_id=self.project_id,
            title="History",
        )
        ConversationMessage.objects.create(
            conversation=self.conversation,
            role=ConversationMessage.Role.USER,
            content="first question",
        )
        self.assistant_1 = ConversationMessage.objects.create(
            conversation=self.conversation,
            role=ConversationMessage.Role.ASSISTANT,
            content="first answer",
            citations_json=[{"file_path": "pkg/a.py", "start_line": 1, "end_line": 10}],
            trace_json={
                "graph_elements": [
                    {"data": {"id": "pkg/a.py", "label": "a.py", "node_type": "file", "file_path": "pkg/a.py"}},
                    {
                        "data": {
                            "id": "pkg/a.py->pkg/b.py",
                            "source": "pkg/a.py",
                            "target": "pkg/b.py",
                            "relation": "imports",
                        }
                    },
                ]
            },
        )
        ConversationMessage.objects.create(
            conversation=self.conversation,
            role=ConversationMessage.Role.USER,
            content="second question",
        )
        self.assistant_2 = ConversationMessage.objects.create(
            conversation=self.conversation,
            role=ConversationMessage.Role.ASSISTANT,
            content="second answer",
            citations_json=[{"file_path": "pkg/c.py", "start_line": 2, "end_line": 20}],
            trace_json={
                "graph_elements": [
                    {"data": {"id": "pkg/c.py", "label": "c.py", "node_type": "file", "file_path": "pkg/c.py"}},
                ]
            },
        )

    def test_query_page_restores_latest_assistant_graph_snapshot(self) -> None:
        response = self.client.get(
            reverse("rag:query_page"),
            {"repo_id": str(self.job.id), "conversation_id": str(self.conversation.id)},
        )
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("pkg/c.py", body)
        self.assertIn('id="graph-empty-state"', body)
        self.assertIn('id="codebase-graph-canvas"', body)
        self.assertIn("graph-empty-state", body)
        self.assertIn("hidden", body)

    def test_query_turn_loads_selected_assistant_turn_graph_and_citations(self) -> None:
        response = self.client.get(
            reverse("rag:query_turn"),
            {
                "repo_id": str(self.job.id),
                "project_id": self.project_id,
                "conversation_id": str(self.conversation.id),
                "message_id": str(self.assistant_1.id),
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("pkg/a.py", body)
        self.assertIn("first answer", body)
