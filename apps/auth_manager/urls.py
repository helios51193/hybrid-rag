from django.urls import path
from . import views

app_name = "auth_manager"

urlpatterns = [path('login', views.user_login, name="auth_login"),
               path('logout', views.user_logout, name="auth_logout"),
               path('signup', views.user_signup, name="auth_signup"),]