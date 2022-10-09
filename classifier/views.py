from email import message
from pyexpat import model
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import pickle
import numpy as np
import pandas as pd
import os
from django.conf import settings
import sklearn

# Create your views here.
def GetForm(request):
    return render(request, 'classifier/index.html')

@require_http_methods(['POST'])
@csrf_exempt
def Predict(request):
    age = request.POST.get('age', 0)
    sex = request.POST.get('sex', 'Male')
    polyuria = request.POST.get('polyuria', 'Yes')
    polydipsia = request.POST.get('polydipsia', 'Yes')
    sudden_weight_loss = request.POST.get('sudden_weight_loss', 'Yes')
    weakness = request.POST.get('weakness', 'Yes')
    polyphagia = request.POST.get('polyphagia', 'Yes')
    genital_thrush = request.POST.get('genital_thrush', 'Yes')
    visual_blurring = request.POST.get('visual_blurring', 'Yes')
    itching = request.POST.get('itching', 'Yes')
    irritability = request.POST.get('irritability', 'Yes')
    delayed_healing = request.POST.get('delayed_healing', 'Yes')
    partial_paresis = request.POST.get('partial_paresis', 'Yes')
    muscle_stiffness = request.POST.get('muscle_stiffness', 'Yes')
    alopecia = request.POST.get('alopecia', 'Yes')
    obesity = request.POST.get('obesity', 'Yes')
    with open(os.path.join(settings.AI_MODEL_DIR, 'model.pickle'), "rb") as input_file:
        model = pickle.load(input_file)
    
    with open(os.path.join(settings.AI_MODEL_DIR, 'preprocess.pickle'), "rb") as input_file:
        preprocess = pickle.load(input_file)
    
    X = pd.DataFrame(
        {
            'Age': [age],
            'Gender': [sex],
            'Polyuria': [polyuria],
            'Polydipsia': [polydipsia],
            'sudden weight loss': [sudden_weight_loss],
            'weakness': [weakness],
            'Polyphagia': [polyphagia],
            'Genital thrush': [genital_thrush],
            'visual blurring': [visual_blurring],
            'Itching': [itching],
            'Irritability': [irritability],
            'delayed healing': [delayed_healing],
            'partial paresis': [partial_paresis],
            'muscle stiffness': [muscle_stiffness],
            'Alopecia': [alopecia],
            'Obesity': [obesity]
        }
    )
    X = preprocess.fit_transform(X)
    result = model.predict(X)
    message_str = {0: 'negative', 1: 'positive'}
    message = "Your result is: "+ message_str[result[0]]
    return HttpResponse(message)

