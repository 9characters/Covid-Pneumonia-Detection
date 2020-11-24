from django.urls import path
from . import views

app_name = 'covidApp'

urlpatterns = [
    path('', views.index, name='index'),
    path('aboutus/', views.aboutus, name='aboutus'),
]
