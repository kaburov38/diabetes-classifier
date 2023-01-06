from email import message
from pyexpat import model
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import pickle
import numpy as np
import pandas as pd
import os
from django.conf import settings
import sklearn

with open(os.path.join(settings.AI_MODEL_DIR, 'model.pickle'), "rb") as input_file:
        model = pickle.load(input_file)
    
with open(os.path.join(settings.AI_MODEL_DIR, 'preprocess.pickle'), "rb") as input_file:
    preprocess = pickle.load(input_file)

# Create your views here.
def GetForm(request):
    return render(request, 'classifier/index.html')

@require_http_methods(['GET'])
@csrf_exempt
def Predict(request):
    age = request.GET.get('age', 0)
    sex = request.GET.get('sex', 'Male')
    polyuria = request.GET.get('polyuria', 'Yes')
    polydipsia = request.GET.get('polydipsia', 'Yes')
    sudden_weight_loss = request.GET.get('sudden_weight_loss', 'Yes')
    weakness = request.GET.get('weakness', 'Yes')
    polyphagia = request.GET.get('polyphagia', 'Yes')
    genital_thrush = request.GET.get('genital_thrush', 'Yes')
    visual_blurring = request.GET.get('visual_blurring', 'Yes')
    itching = request.GET.get('itching', 'Yes')
    irritability = request.GET.get('irritability', 'Yes')
    delayed_healing = request.GET.get('delayed_healing', 'Yes')
    partial_paresis = request.GET.get('partial_paresis', 'Yes')
    muscle_stiffness = request.GET.get('muscle_stiffness', 'Yes')
    alopecia = request.GET.get('alopecia', 'Yes')
    obesity = request.GET.get('obesity', 'Yes')
    
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
    X = preprocess.transform(X)
    result = model.predict(X)
    message_str = {0: 'Negative', 1: 'Positive'}
    return JsonResponse({
        'result':  message_str[result[0]]
    })