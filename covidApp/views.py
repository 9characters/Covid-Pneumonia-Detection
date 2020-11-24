from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K

output_classes = {0: 'Covid', 1: 'Normal', 2: 'Pneumonia'}

# li = list(out_dict.keys())

def index(request):
    if request.method == 'POST':
        if 'myfile' not in request.FILES:
            return HttpResponseRedirect(reverse('index'))

        elif request.FILES['myfile']:

            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            m = str(filename)
            K.clear_session()

            im = Image.open("{}/".format(settings.MEDIA_ROOT) + m)

            # You don't have to change this resolution, it is just to display on the screen
            j = im.resize((256, 256),)
            l = "predicted.jpg"
            j.save("{}/".format(settings.MEDIA_ROOT) + l)
            file_url = fs.url(l)	
            
            model = load_model('covidApp/model.hdf5', compile=False)

            # Change this target_size as per your trained resolution
            img = image.load_img(myfile, target_size=(32, 32))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x).flatten()
            pred_class = np.argmax(pred)
            predicted_category = output_classes[pred_class]

            return render(request, "index.html", {'result': predicted_category, 'file_url': file_url})

    return render(request, "index.html")

def aboutus(request):
    return render(request, 'aboutus.html')