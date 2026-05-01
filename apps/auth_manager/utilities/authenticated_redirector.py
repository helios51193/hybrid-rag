from django.shortcuts import redirect

def redirect_authenticated(to=""):
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if request.user.is_authenticated:
                return redirect(to)
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator