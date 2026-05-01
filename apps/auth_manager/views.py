import traceback
from django.shortcuts import render,redirect

from apps.auth_manager.utilities.authenticated_redirector import redirect_authenticated

from .forms import LoginForm,SignupForm
from django.contrib.auth import authenticate, login, logout
# Create your views here.



@redirect_authenticated(to='document_manager:dashboard')
def user_login(request):

    form = LoginForm()
    context = {
        "login_form":form,
        "has_errors":False,
        "errors":[]
    }
    try:
        if request.method == 'GET':
            return render(request,"auth_manager/login.jinja", context=context)
        if request.method == 'POST':
            
            form = LoginForm(request.POST)
            if form.is_valid():
                
                user = authenticate(request, email=form.cleaned_data["email"], password=form.cleaned_data['password'])
                if user is not None:
                    print("Authenticated")
                    login(request,user)
                    return redirect('document_manager:dashboard')
                else:
                    context['has_errors'] = True
                    context['errors'] = ['Invalid email and/or password']

            else:
                errors = [error for field_errors in form.errors.values() for error in field_errors]
                context['has_errors'] = True
                context['errors'] = errors

            return render(request,"auth_manager/login.jinja", context=context)
    except Exception as e:
        err = traceback.format_exc()
        print(f"{err}")
        context['has_errors'] = True
        context['errors'] = ['Unknows Error Occured, Contact Admin',f"{e}"]

def user_logout(request):
    logout(request)
    return redirect('auth_manager:auth_login')

def user_signup(request):
    
    if request.user.is_authenticated:
        return redirect("document_manager:document_dashboard")
    
    form = SignupForm()
    context = {
        "signup_form":form,
        "has_errors":False,
        "errors":[]
    }

    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect('auth_manager:auth_login')
        else:
            errors = [error for field_errors in form.errors.values() for error in field_errors]
            context['has_errors'] = True
            context['errors'] = errors
        
    return render(request, "auth_manager/signup.jinja", context=context)