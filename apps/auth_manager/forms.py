from django import forms
from django.core.exceptions import ValidationError

from apps.auth_manager.models import User


class LoginForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "class": "input input-bordered w-full",
            "placeholder": "Enter your email"
        })
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            "class": "input input-bordered w-full",
            "placeholder": "Enter your password"
        })
    )

class SignupForm(forms.ModelForm):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "class": "input input-bordered w-full",
            "placeholder": "Enter your email"
        })
    )
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            "class": "input input-bordered w-full",
            "placeholder": "Enter your password"
        })
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={
            "class": "input input-bordered w-full",
            "placeholder": "Confirm your password"
        })
    )

    class Meta:
        model = User
        fields = ["email"]
    
    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email is already taken.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get("password1")
        p2 = cleaned_data.get("password2")

        if p1 and p2 and p1 != p2:
            self.add_error("password2", "Passwords do not match")

        return cleaned_data

    def save(self, commit=True):
        email = self.cleaned_data["email"].lower()

        user = super().save(commit=False)
        user.username = email      # use email as username
        user.email = email
        user.set_password(self.cleaned_data["password1"])

        if commit:
            user.save()
        return user