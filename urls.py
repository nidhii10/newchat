from django.urls import path
from chatbot_app import views
#from django.views.decorators.csrf import csrf_exempt
from .views import chatbot_view
 
urlpatterns = [
    path('', views.index),
    path('chatbot_view/', chatbot_view, name='chatbot_view'),

]