from django.contrib import admin

# Register your models here.
from .models import *

# Register your models here.*
# @admin.register(Park)
# class UserAdmin(admin.ModelAdmin):
#     list_display=('id','noplate')
    
@admin.register(Pi)
class UserAdmin(admin.ModelAdmin):
    list_display=('pic','iii')    

@admin.register(Tasbir)
class UserAdmin(admin.ModelAdmin):
    list_display=('user','tasbir')    

@admin.register(FineDetail)
class FineDetailAdmin(admin.ModelAdmin):
    list_display = ['traffic_officer', 'vehicle_number', 'owner_email', 'owner_name', 'location', 'time', 'fine_types', 'other_fine', 'common_amount', 'fine_reason', 'created_at', 'paid']
    list_filter = ['traffic_officer', 'time', 'created_at', 'paid']
    search_fields = ['vehicle_number', 'owner_email', 'owner_name', 'location', 'fine_types', 'fine_reason']

# @admin.register(Paaark)
# class UserAdmin(admin.ModelAdmin):
#     list_display=('id','noplate','category','email_address')


from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from .models import Paaark

@admin.register(Paaark)
class UserAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'noplate', 'edit_button', 'engine_number', 'chassis_number',
        'insurance_details', 'contact_number', 'email_address', 'owner',
        'color', 'bluebook_expiry_date', 'category'
    )
    search_fields = (
        'noplate', 'modifiednoplate', 'engine_number', 'chassis_number',
        'insurance_details', 'contact_number', 'email_address', 'owner',
        'color', 'category'
    )
    list_filter = ('color', 'bluebook_expiry_date', 'category')

    fieldsets = (
        (None, {
            'fields': (
                'noplate', 'engine_number', 'chassis_number', 
                'insurance_details', 'contact_number', 
                'email_address', 'owner', 'color', 
                'bluebook_expiry_date', 'category'
            )
        }),
        ('Advanced options', {
            'classes': ('collapse',),
            'fields': ('modifiednoplate',)
        }),
    )

    readonly_fields = ('modifiednoplate',)

    def save_model(self, request, obj, form, change):
        # Clean and convert 'noplate' to lowercase before saving in 'modifiednoplate'
        obj.modifiednoplate = ''.join(obj.noplate.split()).lower()
        super().save_model(request, obj, form, change)
    
    def edit_button(self, obj):
        url = reverse('admin:%s_%s_change' % (obj._meta.app_label, obj._meta.model_name),  args=[obj.id])
        return format_html('<a class="button" href="{}">Edit</a>', url)
    
    edit_button.short_description = 'Edit'
    edit_button.allow_tags = True


