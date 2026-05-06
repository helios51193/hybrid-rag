from django.urls import path

from apps.rag import views

app_name = "rag"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("repositories/add/", views.add_repository, name="add_repository"),
    path("repositories/<int:repo_id>/process/", views.process_repository, name="process_repository"),
    path("repositories/<int:repo_id>/", views.delete_repository, name="delete_repository"),
    path("repositories/<int:repo_id>/status-row/", views.repository_status_row, name="repository_status_row"),
    path("query/", views.query_page, name="query_page"),
]
