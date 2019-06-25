from django.http import HttpResponse

# Create your views here.
def index(request):
    return HttpResponse("<h1>Your data map</h1>\n<p>will be here soon</p>")

def single_curve(request):
    return HttpResponse("<h1>Training curve</h1>\n<p>Training curves are not yet prepared for the web app.</p>")

def compare_batches(request):
    return HttpResponse("<h1>Batch comparison</h1>\n<p>Batch comparisons are not yet prepared for the web app.</p>")

