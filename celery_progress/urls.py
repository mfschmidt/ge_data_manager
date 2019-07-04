from django.urls import path
from . import views

app_name = 'celery_progress'
urlpatterns = [
    path('<uuid:task_id>/', views.get_progress, name='task_status')
]
