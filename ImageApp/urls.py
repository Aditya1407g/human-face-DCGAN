from django.urls import path

from . import views

urlpatterns = [
    path("index.html", views.index, name="index"),
    path('AdminLogin.html', views.AdminLogin, name="AdminLogin"), 
    path('AdminLoginAction', views.AdminLoginAction, name="AdminLoginAction"),
    path('TexttoImage', views.TexttoImage, name="TexttoImage"),
    path('TrainModel', views.TrainModel, name="TrainModel"),
    path('TexttoImageAction', views.TexttoImageAction, name="TexttoImageAction"),	   
    path('HumanFaces', views.HumanFaces, name="HumanFaces"),
    path('HumanFacesAction', views.HumanFacesAction, name="HumanFacesAction"),	   
]
