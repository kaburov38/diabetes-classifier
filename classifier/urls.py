from django.urls import path
from . import views

urlpatterns = [
    path('service/predict', views.Predict),
    path('', views.GetForm),
]