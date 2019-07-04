from __future__ import absolute_import, unicode_literals

from django.http import HttpResponse
from django.shortcuts import render
from django.views import generic

from ge_data_manager.celery import celery_id_from_name

from .tasks import clear_jobs, collect_jobs
from .models import PushResult, ResultSummary


def index(request):
    """ The main view, entry point to the site. """

    context={
        'title': 'Gene Expression Main Page',
        'latest_refresh': 'never',
        'n_results': 0,
    }
    if ResultSummary.objects.count() > 1:
        latest_result_summary = ResultSummary.objects.latest('summary_date')
        context['n_results'] = PushResult.objects.count()
        context['latest_result_summary'] = latest_result_summary

    return render(request, 'gedata/index.html', context=context)

class ResultView(generic.DetailView):
    model = PushResult
    template_name = 'gedata/result.html'

class ResultsView(generic.ListView):
    model = PushResult
    template_name = 'gedata/results.html'

def rest_refresh(request):
    """ Execute the celery task to refresh the result list, and return json with the task_id necessary for
        checking on the job periodically and updating the progress bar.
    """
    jobs_id = celery_id_from_name("collect_jobs")
    if jobs_id is None:
        clear_jobs()
        celery_result = collect_jobs.delay("/data", new_only=False)
        jobs_id = celery_result.task_id

    return HttpResponse("{" + "\"task_id\": \"{}\"".format(jobs_id) + "}")

"""
def summarize_bids(request, bids_key, bids_val):
    return HttpResponse("<h1>{} = {} results</h1>\n<p>{} results where {} == {}.</p>".format(
        bids_key, bids_val, 0, bids_key, bids_val
    ))

def result(request):
    the_result = PushResult.objects.latest('end_date')
    return HttpResponse("<h1>A result</h1>\n<p>{}</p>\n<p>{} seconds</p>".format(the_result.json_path, the_result.duration))

def inventory(request):
    return render(request, "gedata/inventory.html", context={'results_list': PushResult.objects.all()})
"""