from django.shortcuts import render
from django.http import HttpResponse
from .utils import generate_text  # importing function from transformerModel.py
from django.http import JsonResponse

# Create your views here.

def index(request):
    return render(request, 'index.html')



def chatbot_view(request):
    if request.method == 'POST':
        sequence = request.POST.get('sequence', '')
        max_len = int(request.POST.get('max_len', 50))
        model_path = "C:/Users/Nidhi/Downloads/newchat/chatbot_project/model"
        generated_text = generate_text(model_path, sequence, max_len)

        # Return a JSON response with the generated text
        return JsonResponse({'generated_text': generated_text})

    # If it's a GET request, render the chat.html template
    return render(request, 'index.html', {'sequence': '', 'generated_text': ''})

'''def chatbot_view(request):
    if request.method == 'POST':
        sequence = request.POST.get('sequence', '')
        max_len = int(request.POST.get('max_len', 50))
        model_path = "D:/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/model"
        
        # Debug prints
        print(f"Received sequence: {sequence}, max_len: {max_len}")
        
        generated_text = generate_text(model_path, sequence, max_len)
        
        # Debug print
        print(f"Generated text: {generated_text}")
        
        return render(request, 'chat.html', {'sequence': sequence, 'generated_text': generated_text})

    return render(request, 'chat.html', {'sequence': '', 'generated_text': ''})'''
