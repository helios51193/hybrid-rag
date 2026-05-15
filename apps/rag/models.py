from django.db import models


class CodeNode(models.Model):
    indexing_job = models.ForeignKey(
        "IndexingJob",
        on_delete=models.CASCADE,
        related_name="code_nodes",
        null=True,
        blank=True,
    )
    project_id = models.CharField(max_length=255, db_index=True)
    node_id = models.CharField(max_length=1024)
    node_type = models.CharField(max_length=64, default="file")
    language = models.CharField(max_length=64, blank=True, default="")
    file_path = models.CharField(max_length=2048, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("project_id", "node_id")
        indexes = [
            models.Index(fields=["project_id", "node_type"]),
        ]


class CodeEdge(models.Model):
    indexing_job = models.ForeignKey(
        "IndexingJob",
        on_delete=models.CASCADE,
        related_name="code_edges",
        null=True,
        blank=True,
    )
    project_id = models.CharField(max_length=255, db_index=True)
    source_node_id = models.CharField(max_length=1024)
    target_node_id = models.CharField(max_length=1024)
    relation = models.CharField(max_length=64, default="imports")
    weight = models.FloatField(default=1.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("project_id", "source_node_id", "target_node_id", "relation")
        indexes = [
            models.Index(fields=["project_id", "source_node_id"]),
            models.Index(fields=["project_id", "target_node_id"]),
        ]

class IndexingJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        RUNNING = "RUNNING", "Running"
        DONE = "DONE", "Done"
        FAILED = "FAILED", "Failed"

    project_id = models.CharField(max_length=255, db_index=True)
    source_dir = models.CharField(max_length=2048)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )

    # Optional task correlation
    task_id = models.CharField(max_length=255, blank=True, default="", db_index=True)

    # Timing
    queued_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    # Result stats
    documents_collected = models.PositiveIntegerField(default=0)
    chunks_created = models.PositiveIntegerField(default=0)
    vectors_upserted = models.PositiveIntegerField(default=0)
    graph_nodes = models.PositiveIntegerField(default=0)
    graph_edges = models.PositiveIntegerField(default=0)
    duration_seconds = models.FloatField(default=0.0)

    # Error/debug
    error_message = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["project_id", "status"]),
            models.Index(fields=["queued_at"]),
        ]
        ordering = ["-queued_at"]

    def __str__(self) -> str:
        return f"{self.project_id} [{self.status}] #{self.id}"


class Conversation(models.Model):
    project_id = models.CharField(max_length=255, db_index=True)
    title = models.CharField(max_length=255, blank=True, default="")
    is_archived = models.BooleanField(default=False, db_index=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["project_id", "is_archived"]),
            models.Index(fields=["-updated_at"]),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.project_id} :: {self.title or self.id}"


class ConversationMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=16, choices=Role.choices, db_index=True)
    content = models.TextField()
    citations_json = models.JSONField(default=list, blank=True)
    trace_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
            models.Index(fields=["conversation", "role", "created_at"]),
        ]
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.conversation_id}:{self.role}:{self.id}"
