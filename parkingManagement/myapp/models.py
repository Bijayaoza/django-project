from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class Park(models.Model):
    noplate = models.CharField(max_length=50, blank=False, null=False)
    modifiednoplate = models.CharField(max_length=50, blank=True, null=True)
    engine_number = models.CharField(max_length=20, blank=False, null=False)
    chassis_number = models.CharField(max_length=20, blank=False, null=False)
    insurance_details = models.TextField(blank=False, null=False)
    contact_number = models.CharField(max_length=15, blank=False, null=False)
    email_address = models.EmailField(blank=False, null=False)
    car_owner = models.CharField(max_length=100, blank=False, null=False)
    car_color = models.CharField(max_length=50, blank=False, null=False)
    bluebook_expiry_date = models.DateField(blank=False, null=False)
    # Add other fields as needed
    def save(self, *args, **kwargs):
        # Save the original 'noplate' data
        super(Park, self).save(*args, **kwargs)

        # Clean and convert 'noplate' to lowercase before saving in 'modifiednoplate'
        self.modifiednoplate = ''.join(self.noplate.split()).lower()
        super(Park, self).save(*args, **kwargs)


    def __str__(self):
        return self.noplate   


class Pi(models.Model):
    pic=models.ImageField(null=True,blank=True)
    iii = models.CharField(max_length=50, blank=True, null=True)

class Tasbir(models.Model):
    user=models.OneToOneField(User, on_delete=models.CASCADE)
    tasbir=models.ImageField(null=True,blank=True)
    

class FineDetail(models.Model):
    traffic_officer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    vehicle_number = models.CharField(max_length=100)
    owner_email = models.EmailField(max_length=100)
    owner_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    time = models.TimeField()
    fine_types = models.CharField(max_length=255)  # Storing fine types as comma-separated string
    other_fine = models.TextField(blank=True)
    common_amount = models.DecimalField(max_digits=10, decimal_places=2,blank=True,null=True)
    fine_reason = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    paid = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.vehicle_number} - {self.owner_name}"

  

class Paaark(models.Model):
    noplate = models.CharField(max_length=50, blank=False, null=False)
    modifiednoplate = models.CharField(max_length=50, blank=True, null=True)
    engine_number = models.CharField(max_length=20, blank=False, null=False)
    chassis_number = models.CharField(max_length=20, blank=False, null=False)
    insurance_details = models.TextField(blank=False, null=False)
    contact_number = models.CharField(max_length=15, blank=False, null=False)
    email_address = models.EmailField(blank=False, null=False)
    owner = models.CharField(max_length=100, blank=False, null=False)
    color = models.CharField(max_length=50, blank=False, null=False)
    bluebook_expiry_date = models.DateField(blank=False, null=False)
    category=models.CharField(max_length=1, blank=False, null=False)
    
    class Meta:
        verbose_name = "Licence Plate Detail"
        verbose_name_plural = "Licence Plate Details"
    
    # Add other fields as needed
    def save(self, *args, **kwargs):
        # Save the original 'noplate' data
        super(Paaark, self).save(*args, **kwargs) 

        # Clean and convert 'noplate' to lowercase before saving in 'modifiednoplate'
        self.modifiednoplate = ''.join(self.noplate.split()).lower()
        super(Paaark, self).save(*args, **kwargs)


    def __str__(self):
        return self.noplate  