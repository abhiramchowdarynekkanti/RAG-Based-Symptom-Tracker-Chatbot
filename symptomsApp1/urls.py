from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Register.html", views.Register, name="Register"),
		     path("RegisterAction", views.RegisterAction, name="RegisterAction"),
		     path("Chatbot", views.Chatbot, name="Chatbot"),
		     path("ChatData", views.ChatData, name="ChatData"),	
		     path("UserScreen", views.UserScreen, name="UserScreen"),
		     path("ViewLog", views.ViewLog, name="ViewLog"),	
		    ]