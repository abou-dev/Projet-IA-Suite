from django.urls import path
from .views import predict, result, home

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('result/', result, name='result'),
    path('', home, name='home'),
]
