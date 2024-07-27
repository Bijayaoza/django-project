import base64
from django.core.files.base import ContentFile
import uuid
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.
from .models import *
from .forms import *
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.views import View
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group
from .decorators import unauthenticated_user, allowed_users, admin_only
from django.http import JsonResponse
from .forms import FineTypesForm
from .models import FineDetail
from django.utils import timezone
import requests
import json
from django.http import HttpResponseServerError
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
# from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from django.urls import reverse


import re
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from easyocr import Reader


from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings

from django.contrib import messages
from django.http import HttpResponseRedirect
from .forms import CreateUserForm
from django.contrib.auth.models import Group
from django.contrib.auth.models import User


######## paractice###########
# from keras.models import load_model
# from keras.preprocessing import image
# import tensorflow as tf
# import json
# from tensorflow import Graph, Session


# img_height, img_width=224,224
# with open('./models/imagenet_classes.json','r') as f:
#     labelInfo=f.read()

# labelInfo=json.loads(labelInfo)


# model_graph = Graph()
# with model_graph.as_default():
#     tf_session = Session()
#     with tf_session.as_default():
#         model=load_model('./models/MobileNetModelImagenet.h5')


# @login_required
# def predict(request):
#     if request.method == 'POST':
#         user_profile = Tasbir.objects.get(user=request.user)
#         if user_profile.tasbir:
#             file_path = user_profile.tasbir.path
#             img = cv2.imread(file_path)

#             grayConvertedImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             filteredImg = cv2.bilateralFilter(grayConvertedImg, 11, 17, 17)
#             edgedImg = cv2.Canny(filteredImg, 30, 200)

#             keyPoints = cv2.findContours(edgedImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours = imutils.grab_contours(keyPoints)
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#             location = None

#             for contour in contours:
#                 approx = cv2.approxPolyDP(contour, 10, True)
#                 if len(approx) == 4:
#                     location = approx
#                     break  # Exit the loop after finding the first contour

#             if location is not None:
#                 mask = np.zeros(grayConvertedImg.shape, np.uint8)
#                 newImage = cv2.drawContours(mask, [location], 0, 255, -1)
#                 newImage = cv2.bitwise_and(img, img, mask=mask)

#                 (x, y) = np.where(mask == 255)
#                 (x1, y1) = (np.min(x), np.min(y))
#                 (x2, y2) = (np.max(x), np.max(y))
#                 croppedNumberPlate = grayConvertedImg[x1:x2, y1:y2]

#                 reader = Reader(['en'])
#                 result = reader.readtext(croppedNumberPlate)

#                 predicted_text = ''
#                 confidence_levels = []

#                 for detection in result:
#                     text, confidence = detection[1], detection[2]
#                     predicted_text += text + ' '
#                     confidence_levels.append(confidence)

#                 print("OCR Result:", predicted_text)
#                 print("Confidence Levels:", confidence_levels)

#                 context = {'predictedLabel': predicted_text, 'confidenceLevels': confidence_levels}
#                                             # Get the path to the image file
#                 image_path = user_profile.tasbir.path

#                 # Delete the image from the static file
#                 if os.path.exists(image_path):
#                     os.remove(image_path)

#                 # Delete the entire Tasbir instance (including the user reference)
#                 user_profile.delete()
#                 return render(request, 'myapp/parking.html', context)
#             else:
#                 print("No contours found.")
#                 #  return render(request,'myapp/parking.html','msseg':'no con')

#     return redirect('delete_photo')


def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.8, ocr_conf_threshold=0.5, txt_only=False):
    def apply_filters(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    # Load labels
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if the input tensor is float type
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize OCR reader
    reader = easyocr.Reader(['en'])

    # Read and preprocess the image
    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image_rgb.shape

    input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    image_resized = cv2.resize(image_rgb, input_shape)

    if float_input:
        image_resized = (np.float32(image_resized) - input_mean) / input_std

    # Set the tensor to the model and invoke
    interpreter.set_tensor(
        input_details[0]['index'], np.expand_dims(image_resized, axis=0))
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    ocr_results_list = []
    ocr_confidences_list = []
    tensorflow_results_list = []

    # Filter results based on confidence score
    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):  # Strictly greater than 0.8
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            if ymin >= ymax or xmin >= xmax:
                continue

            # Region of interest
            roi = image[ymin:ymax, xmin:xmax]
            processed_roi = apply_filters(roi)

            # Perform OCR on the ROI
            ocr_results = reader.readtext(processed_roi)

            # Filter OCR results based on confidence threshold
            filtered_ocr_results = [
                result for result in ocr_results if result[2] > ocr_conf_threshold]

            # Extract and store OCR text and confidences
            ocr_results_list.extend([result[1]
                                    for result in filtered_ocr_results])
            ocr_confidences_list.extend([result[2]
                                        for result in filtered_ocr_results])

            tensorflow_results_list.append({
                'object': labels[int(classes[i])],
                'confidence': scores[i]
            })

    # Remove 'nep' (case-insensitive) from the OCR results
    ocr_results_list = [
        text for text in ocr_results_list if 'nep' not in text.lower()]

    return ocr_results_list, ocr_confidences_list, tensorflow_results_list


@login_required
def predict(request):
    user_profile = Tasbir.objects.get(user=request.user)
    if user_profile.tasbir:
        # Get the path to the image file
        file_path = user_profile.tasbir.path

        # Paths to the model and label file
        PATH_TO_MODEL = 'detect.tflite'
        PATH_TO_LABELS = 'labelmap.pbtxt'
        min_conf_threshold = 0.8  # Set the threshold to 0.8
        ocr_conf_threshold = 0.5  # Set OCR confidence threshold

        if os.path.exists(PATH_TO_MODEL) and os.path.exists(PATH_TO_LABELS):
            # Call the license plate detection function
            ocr_results, ocr_confidences, tensorflow_results = tflite_detect_images(
                PATH_TO_MODEL, file_path, PATH_TO_LABELS, min_conf=min_conf_threshold, ocr_conf_threshold=ocr_conf_threshold
            )

            if not ocr_results:
                context = {
                    'ocrText': 'boundary not detected',
                    'ocrConfidences': ocr_confidences,
                    'tensorflowResults': tensorflow_results
                }
                return render(request, 'myapp/ocr_result.html', context)

            # Prepare data to pass to the template
            context = {
                'ocrText': ' '.join(ocr_results),
                'ocrConfidences': ocr_confidences,
                'tensorflowResults': tensorflow_results
            }

            # Once you have the results, you can render a template with the detected information
            return render(request, 'myapp/ocr_result.html', context)
        else:
            return HttpResponse("Model or label file not found. Please check the file paths.")
    else:
        return redirect('delete_photo')


#####################
@login_required(login_url='login')
def home(request):

    return render(request, 'myapp/parking.html', {'name': 'hello'})


def registration(request):
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')

            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already taken!')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email already registered!')
            else:
                user = form.save()
                group = Group.objects.get(name='staff')
                user.groups.add(group)
                messages.success(
                    request, 'Staff registered successfully: ' + username)
                return HttpResponseRedirect('/sucess/reg/')
        else:
            messages.error(
                request, 'Passwords do not match or other form errors!')

    else:
        form = CreateUserForm()

    context = {'form': form}
    return render(request, 'myapp/garvage.html', context)


@unauthenticated_user  # user login xa vane feri form dekhaunna##decorator ma define xa
def userLogin(request):
    if request.method == "POST":
        fm = AuthenticationForm(request=request, data=request.POST)
        if fm.is_valid():
            print('hiiiiiii')
            uname = fm.cleaned_data['username']
            upass = fm.cleaned_data['password']
            User = authenticate(username=uname, password=upass)
            if User is not None:  # yedi user  ko user  name ra password mily0 vane user  ma hal ani login ma request gar
                login(request, User)
                print(uname)
                return redirect('/profile/')
            else:
                messages.error(request, 'Invalid username or password')
        else:
            # messages.error(request, 'Invalid form submission. Please check the form data.')
            messages.error(request, 'Invalid username or password')
    else:
        fm = AuthenticationForm()
    return render(request, 'myapp/login.html', {'form': fm})


@login_required(login_url='login')
def search_view(request):
    if request.method == 'POST':
        # Process the form data if the request method is POST
        form = SearchForm(request.POST)

        if form.is_valid():
            # If the form is valid, handle the form data
            vehicle_no = form.cleaned_data['vehicle_no']

            if len(vehicle_no) < 8:
                # If the vehicle number is too short, add an error to the form
                form.add_error(
                    'vehicle_no', 'Vehicle number must be at least 8 characters long.')
            else:
                # Using __iexact for case-insensitive exact match
                clean_vehicle_no = ''.join(vehicle_no.split()).lower()

                try:
                    # Use Park.objects.get and handle the DoesNotExist exception
                    park_entry = Paaark.objects.get(
                        modifiednoplate__iexact=clean_vehicle_no)
                    print(park_entry)
                    # Render the search results
                    return render(request, 'myapp/search.html', {'form': form, 'original_vehicle_no': vehicle_no, 'park_entry': park_entry})
                except Paaark.DoesNotExist:
                    # If no matching entry is found, add an error to the form
                    form.add_error('vehicle_no', 'No matching entry found.')
        else:
            # If the form is not valid, render the form with error messages
            return render(request, 'myapp/search.html', {'form': form})

    else:
        # If the request method is GET, create a new form instance and render the form
        form = SearchForm()

    return render(request, 'myapp/search.html', {'form': form})


# yesla chai function vitra function kasari call garna idea dinxa both upload and delete function you super funciton ho
# @login_required(login_url='login')
# def account_settings(request):
#     user = request.user
#     tasbir_form = TasbirForm()

#     if request.method == 'POST':
#         if request.POST.get('upload_photo'):
#             update_photo(request, user, tasbir_form)
#         elif request.POST.get('delete_photo'):
#             delete_photo(request, user)

#     return render(request, 'myapp/tasbir.html', {'user': user, 'tasbir_form': tasbir_form})


def upload_photo(request):
    user = request.user
    error_message = None
    is_admin = request.user.groups.filter(name='admin').exists()
    print(is_admin)
    is_staff = request.user.groups.filter(name='staff').exists()
    print(is_staff)
    # Other context variables

    # Retrieve any error message passed from delete_photo function
    error_message = messages.get_messages(request)

    if request.method == 'POST':
        tasbir_form = TasbirForm(request.POST, request.FILES)

        if tasbir_form.is_valid():
            # Delete existing photo if it exists
            if hasattr(user, 'tasbir') and user.tasbir:
                user.tasbir.delete()
            # Save new photo
            tasbir = tasbir_form.save(commit=False)
            tasbir.user = user
            tasbir.save()
    else:
        # Pass the error message as initial data to the form
        tasbir_form = TasbirForm(initial={'error_message': error_message})

    return render(request, 'myapp/tasbir.html', {'user': user, 'tasbir_form': tasbir_form, 'is_admin': is_admin, 'is_staff': is_staff})


# @login_required
# def delete_photo(request):
#     # if request.method == 'POST':
#     user_profile = Tasbir.objects.get(user=request.user)

#     # Check if the user has a photo before attempting to delete
#     if user_profile.tasbir:
#         # Get the path to the image file
#         image_path = user_profile.tasbir.path

#         # Delete the image from the static file
#         if os.path.exists(image_path):
#             os.remove(image_path)

#         # Delete the entire Tasbir instance (including the user reference)
#         user_profile.delete()
#     return redirect('upload_photo')


@login_required(login_url='login')
@admin_only
def delete_photo(request):
    try:
        user_profile = Tasbir.objects.get(user=request.user)
        if user_profile.tasbir:
            image_path = user_profile.tasbir.path
            if os.path.exists(image_path):
                os.remove(image_path)
            user_profile.delete()
        else:
            # If no photo found to delete, pass error message to upload_photo
            messages.error(
                request, "Error occurred: No photo found to delete.")
    except Tasbir.DoesNotExist:
        # If Tasbir object does not exist, pass error message to upload_photo
        messages.error(request, "Error occurred: No photo found to delete.")

    return redirect('upload_photo')


def fine_types(request):
    if request.method == 'POST':
        vehicle_number = request.POST.get('vehicle_number')
        owner_email = request.POST.get('owner_email')
        owner_name = request.POST.get('owner_name')

        # Store the values in the session
        request.session['vehicle_number'] = vehicle_number
        request.session['owner_email'] = owner_email
        request.session['owner_name'] = owner_name

        # Redirect to the confirm_fines view
        return redirect('confirm_fines')
    else:
        return redirect('delete_photo')


@login_required
def confirm_fines(request):
    # Retrieve session variables
    vehicle_number = request.session.get('vehicle_number')
    owner_email = request.session.get('owner_email')
    owner_name = request.session.get('owner_name')
    traffic_officer_user = request.user
    time = timezone.now()
    print(time)

    ip = requests.get('https://api.ipify.org?format=json')
    ip_data = json.loads(ip.text)
    print(ip_data)
    res = requests.get('http://ip-api.com/json/'+ip_data["ip"])
    location_data_one = res.text
    location_data = json.loads(location_data_one)
    country = location_data['city']
    print(country)

    if request.method == 'POST':
        form = FineTypesForm(request.POST)
        if form.is_valid():
            location = form.cleaned_data['location']
            time = form.cleaned_data['time']
            fine_types = form.cleaned_data.get('fine_types', [])
            if not time:
                # If 'time' is not provided, set it to the current time
                time = timezone.now()

            other_fine = form.cleaned_data.get('other_fine', '')
            common_amount = form.cleaned_data.get('common_amount', '')
            fine_reason = form.cleaned_data.get('fine_reason', '')

            # Save fine details with associated traffic officer
            fine_detail = FineDetail.objects.create(
                traffic_officer=traffic_officer_user,
                vehicle_number=vehicle_number,
                owner_email=owner_email,
                owner_name=owner_name,
                location=location,
                time=time,
                fine_types=fine_types,
                other_fine=other_fine,
                common_amount=common_amount,
                fine_reason=fine_reason
            )

            # Prepare the context for the email template
            context = {
                'owner_name': owner_name,
                'vehicle_number': vehicle_number,
                'location': location,
                'time': time,
                'fine_types': fine_types,
                'other_fine': other_fine,
                'common_amount': common_amount,
                'fine_reason': fine_reason,
                'traffic_officer': traffic_officer_user,

            }

            # Render the email template with context
            email_message = render_to_string('myapp/fine_email.html', context)

            # Send email to the owner using configured sender email address
            send_mail(
                'Fine Details',  # Subject
                email_message,    # Email body
                settings.DEFAULT_FROM_EMAIL,  # Sender's email address
                [owner_email],    # Recipient's email address, should be a list or tuple
                html_message=email_message  # Optional: Email body in HTML format
            )

            # Delete session variables after saving fine details and sending email
            del request.session['vehicle_number']
            del request.session['owner_email']
            del request.session['owner_name']

            # Optionally, you can render a confirmation page
            return render(request, 'myapp/fine_confirmation.html', {'fine_detail': fine_detail})
    else:
        # Populate initial data in the form
        initial_values = {
            'vehicle_number': vehicle_number,
            'owner_email': owner_email,
            'time': time,
            # Add more initial values as needed
        }
        form = FineTypesForm(initial=initial_values)

    # Render the template with the form
    return render(request, 'myapp/choose_fine_types.html', {'form': form})


def fine_detail(request):
    fines = FineDetail.objects.all()
    return render(request, 'myapp/fine_detail.html', {'fines': fines})


class Cfunna(View):
    template_name = ''  # here blank_string is given, as from url.py template url location is passed #note variable name should be written mendatory

    def get(self, request):
        title = 'class based reusable component'
        return render(request, self.template_name, {'title': title})


@login_required(login_url='login')
@admin_only
def park(request):
    return redirect('upload_photo')


@login_required(login_url='login')
# @allowed_users(allowed_roles=['admin'])
@admin_only
def headquater(request):
    return render(request, 'myapp/admin.html')


@login_required(login_url='login')
def staff(request):
    return render(request, 'myapp/staff.html')


# @login_required(login_url='login')
# @allowed_users(allowed_roles=['transport'])
# def transport(request):
#     print('hello')
#     return render(request, 'myapp/transport.html')


def userLogout(request):
    logout(request)
    return redirect('homePage')


# @login_required(login_url='login')
# def update_photo(request):
#     user = request.user
#     if request.method == 'POST':
#         tasbir_form = TasbirForm(request.POST, request.FILES)

#         if tasbir_form.is_valid():
#             # Delete existing photo if it exists
#             if hasattr(user, 'tasbir') and user.tasbir:
#                 user.tasbir.delete()
#             # Save new photo
#             tasbir = tasbir_form.save(commit=False)
#             tasbir.user = user
#             tasbir.save()
#     else:

#     return redirect('account_settings')


@allowed_users(allowed_roles=['transport'])
def add_park(request):
    if request.method == 'POST':
        form = PaaarkForm(request.POST)
        if form.is_valid():
            vehicle_no = form.cleaned_data['noplate']

            if len(vehicle_no) < 8:
                # If the vehicle number is too short, add an error to the form
                form.add_error(
                    'noplate', 'Vehicle number must be at least 8 characters long.')
            else:
                # Using __iexact for case-insensitive exact match
                clean_vehicle_no = ''.join(vehicle_no.split()).lower()
                try:
                    # Use Park.DoesNotExist to catch the exception
                    park_entry = Paaark.objects.get(
                        modifiednoplate__iexact=clean_vehicle_no)
                    form.add_error(
                        'noplate', 'Vehicle plate is already registered.')
                except Paaark.DoesNotExist:
                    form.save()
                    messages.success(
                        request, 'Vehicle details registered successfully.')
                    # Redirect to search results page after saving
                    return redirect('transport')
    else:
        form = PaaarkForm()

    return render(request, 'myapp/transport/add_park.html', {'form': form})


def webcam(request):
    return render(request, 'myapp/webcam.html')


@login_required
def capture_image(request):
    user = request.user
    error_message = None
    is_admin = request.user.groups.filter(name='admin').exists()
    print(is_admin)
    is_staff = request.user.groups.filter(name='staff').exists()
    print(is_staff)

    if request.method == 'POST':
        image_data = request.POST.get('image')
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        data = ContentFile(base64.b64decode(imgstr),
                           name=f'{uuid.uuid4()}.{ext}')

        # Initialize the form with POST data
        tasbir_form = TasbirForm(request.POST, request.FILES)

        if tasbir_form.is_valid():
            # Delete existing photo if it exists
            if hasattr(request.user, 'tasbir') and request.user.tasbir.tasbir:
                request.user.tasbir.tasbir.delete()

            # Save new photo
            tasbir = tasbir_form.save(commit=False)
            tasbir.user = request.user
            tasbir.tasbir = data
            tasbir.save()

            # Print the path for debugging
            print("Image saved at:", tasbir.tasbir.url)
            print("User is:", request.user.username)

            # Render the success page with the image URL
            return render(request, 'myapp/tasbir.html', {
                'user': user,
                'tasbir_form': tasbir_form,
                'is_admin': is_admin,
                'is_staff': is_staff,
                'image_url': tasbir.tasbir.url,
            })

    else:
        # Initialize the form for GET request
        tasbir_form = TasbirForm()

    return render(request, 'myapp/webcam.html', {
        'form': tasbir_form,
        'is_admin': is_admin,
        'is_staff': is_staff,
    })
