from jinja2 import Environment
from django.templatetags.static import static
from django.urls import reverse
from django_jinja.builtins.extensions import CsrfExtension


def environment(**options):

    env = Environment(**options)
    env.globals.update({
        'static': static,
        'url': reverse,
        
    })
    return env