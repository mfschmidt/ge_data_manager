from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('single_curve/', views.single_curve, name='single_curve'),
    path('compare_batches/', views.compare_batches, name='compare_batches'),
]