from django.urls import path

from . import views, forms

app_name = "gedata"
urlpatterns = [
    path('', views.index, name='index'),
    path('REST/refresh/<str:job_name>', views.rest_refresh, name='refresh'),
    path('results/', views.ResultsView.as_view(), name='results'),
    path('result/<int:pk>', views.ResultView.as_view(), name='result'),
    path('filter/', forms.filter_results, name='filter'),
    path('compare/', forms.compare_results, name='compare'),
]

"""
path('inventory/', views.inventory, name='inventory'),
path('<str:bids_key>/<str:bids_val>/', views.summarize_bids, name="summarize_bids"),
path('<str:bids_key1>/<str:bids_val1>/<str:bids_key2>/<str:bids_val2>/', views.summarize_bids, name="summarize_bids"),
"""