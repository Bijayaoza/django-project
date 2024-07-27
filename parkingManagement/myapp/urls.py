from django.contrib import admin
from django.urls import path,include
from . import views
from django.contrib.auth import views as auth_views

admin.site.index_title="Traffic manager"
admin.site.site_header="Transportation Admin"

urlpatterns = [
    path('',views.userLogin,name='homePage'),
    
    path('reg/',views.registration,name='register'),
    path('userlogin/',views.userLogin,name='login'),
    # path('profile/',views.Cfunna.as_view(template_name=''),name='common'),#eg class based
    path('profile/',views.park,name='profilee'),
    path('headquater/',views.headquater,name='headquater'),
    path('staff/',views.staff,name='stafff'),
    # path('transport/',views.transport,name='transport'),
    path('logout/',views.userLogout,name='logout'),

    path('search/',views.search_view, name='search_view'),
    # path('searchveh/', views.search_vehicle, name='fun'),
    # path('account/settings/', views.account_settings, name='account_settings'),
    path('delete-photo/', views.delete_photo, name='delete_photo'),
    path('upload-photo/', views.upload_photo, name='upload_photo'),
    path('predict/', views.predict, name='predict_photo'),
    # path('accounts/',include('django.contrib.auth.urls')),
    path('reset_password/', auth_views.PasswordResetView.as_view(template_name='myapp/password_reset.html'), name='reset_password'),
    path('reset_password_sent/', auth_views.PasswordResetDoneView.as_view(template_name='myapp/password_reset_sent.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='myapp/password_reset_form.html'), name='password_reset_confirm'),
    path('reset_password/complete/', auth_views.PasswordResetCompleteView.as_view(template_name='myapp/password_reset_done.html'), name='password_reset_complete'),
    

    # path('issue_fine/', views.issue_fine, name='issue_fine'),
    path('choose_fine_types/', views.fine_types, name='choose_fine_types'),
    path('confirm_fines/', views.confirm_fines, name='confirm_fines'),
    path('fine/', views.fine_detail, name='fine_detail'),
    # path('edit/<str:vehicle_number>/', views.add_edit_park, name='transport_edit'),
    path('edit/', views.add_park, name='transport'),  # New URL pattern without the vehicle_number parameter
    # path('webcam/', views.webcam, name='webcam'),
    path('webcam/', views.webcam, name='webcam'),
    path('capture-image/', views.capture_image, name='capture_image'),
    # path('save-image/', views.save_image, name='save_image'),  # Add this line



]


# path('',views.reg,name='homepage'), 
#     path('reg/',views.reg,name='registration'),   
#     path('delete/<int:id>/',views.delete,name="deleteid"),
#     path('<int:id>/',views.update,name="update"),


    ###for class based passing url##### note:in class based view object is passed from argument but at funct based view list is used ie key and value pair
    # path('cfunna',views.Cfunna.as_view(template_name='myapp/funna.html'),name='cfunna'),
    # path('cflora',views.Cfunna.as_view(template_name='myapp/flora.html'),name='cflora'),

