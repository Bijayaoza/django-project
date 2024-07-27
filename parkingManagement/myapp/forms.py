from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django import forms
from .models import Tasbir,Paaark
from django.contrib.auth.models import User
# from .models import UserProfile


class CreateUserForm(UserCreationForm):
    #in custom i could not add placeholder so i did manually
    password1 = forms.CharField(
        label="Password",
        strip=False,
        widget=forms.PasswordInput(attrs={'placeholder': 'Password'}),
        required=True,
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password'}),
        strip=False,
        required=True,
    )
    #in buit_in(custom) required cannot be included so i have done it manually
    email = forms.EmailField(
        max_length=254,
        required=True,  # Make email field required
        widget=forms.EmailInput(attrs={'placeholder': 'Email'}),
    )

    class Meta:
        model=User
        fields=['username','email','password1','password2']
        widgets = {
            'username': forms.TextInput(attrs={'placeholder': 'Username'}),
        
        }



class signupform(UserCreationForm):
    password2=forms.CharField(label='confirm password',widget=forms.PasswordInput)
    class Meta:
        model=User
        fields=['username','first_name','last_name','email']#name haru yehi format ma hunu porxa yo direct  libary batw imort garya ho so..
        label={'email':'Email'}

class SignInForm(UserCreationForm):
    class Meta:
        model=User
        fields=['username','password']

class SearchForm(forms.Form):
    vehicle_no = forms.CharField(
        label="Vehicle No",
        required=True,
        min_length=8,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Vehicle No'}),
    )


# class UserProfileForm(forms.ModelForm):
#     class Meta:
#         model = UserProfile
#         fields = ['bio', 'picture']
    
class TasbirForm(ModelForm):
    class Meta:
        model=Tasbir
        fields=['tasbir']
        labels={'tasbir':''}

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
    




class FineTypesForm(forms.Form):
    vehicle_number = forms.CharField(label='Vehicle Number')
    owner_email = forms.EmailField(label='Owner Email')
    location = forms.CharField(label='Location', required=False)
    time = forms.TimeField(label='Time', required=False)
    fine_types = forms.MultipleChoiceField(
        label='Select Fine Types',
        widget=forms.CheckboxSelectMultiple,
        choices=[
            ('speeding', 'Speeding'),
            ('illegal_parking', 'Illegal Parking'),
            ('running_red_light', 'Running Red Light'),
            # Add more fine types as needed
        ],
        required=False
    )
    fine_reason = forms.CharField(label='Fine Reason', required=False)  # Add fine reason field
    other_fine = forms.CharField(label='Other Fine Types', widget=forms.Textarea, required=False)
    common_amount = forms.DecimalField(label='Common Amount', required=False)

    # Add more fields as needed for reasons of other fine types

    def clean(self):
        cleaned_data = super().clean()
        fine_types = cleaned_data.get('fine_types', [])
        other_fine = cleaned_data.get('other_fine', '')

        # Check if no fine types are selected and other fine is empty
        if not fine_types and not other_fine:
            # If both are empty, raise validation error with a custom error message
            raise forms.ValidationError("Please select at least one fine type or provide other fine details.")

        # Return the cleaned data
        return cleaned_data

class PaaarkForm(forms.ModelForm):
    class Meta:
        model = Paaark
        fields = '__all__'  # Include all fields from the Park model  
        exclude = ['modifiednoplate']
              