from django.db import models


class CodeNode(models.Model):
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
