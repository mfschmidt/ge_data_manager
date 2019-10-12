from __future__ import absolute_import, unicode_literals

from django.http import HttpResponse
from django.shortcuts import render
from django.views import generic

from ge_data_manager.celery import celery_id_from_name, celery_plots_in_progress

from .tasks import clear_jobs, collect_jobs, build_plot, assess_performance, interpret_descriptor
from .models import PushResult, ResultSummary


def index(request):
    """ The main view, entry point to the site. """

    context={
        'title': 'Gene Expression Main Page',
        'latest_result_summary': ResultSummary.empty(),
    }
    # if ResultSummary.objects.count() > 0:
    #     context['latest_result_summary'] = ResultSummary.objects.latest('summary_date')

    return render(request, 'gedata/index.html', context=context)

class ResultView(generic.DetailView):
    model = PushResult
    template_name = 'gedata/result.html'
    # Additional data for the footer
    def get_context_data(self, **kwargs):
        context = super(ResultView, self).get_context_data(**kwargs)
        context['n_results'] = PushResult.objects.count()
        context['m_results'] = ResultSummary.objects.latest('summary_date').num_results
        # context['latest_result_summary'] = ResultSummary.objects.latest('summary_date')
        return context

class ResultsView(generic.ListView):
    model = PushResult
    template_name = 'gedata/results.html'
    # Additional data for the footer
    def get_context_data(self, **kwargs):
        context = super(ResultsView, self).get_context_data(**kwargs)
        context['n_results'] = PushResult.objects.count()
        context['m_results'] = ResultSummary.objects.latest('summary_date').num_results
        context['latest_result_summary'] = ResultSummary.objects.latest('summary_date')
        return context

    def get_queryset(self):
        return PushResult.objects.filter(shuffle='derivatives')

def rest_inventory(request, signature):
    """ From an inventory id, like 'hcpww00s', return four-part inventory json. """

    comp, parby, splby, mask, algo, phase, opposite_phase, relevant_results_queryset = interpret_descriptor(signature)
    sign_string = "\"{}\":\"{}\"".format("signature", signature)
    none_string = "\"{}\":\"{}\"".format("none", len(relevant_results_queryset.filter(shuffle="derivatives")))
    agno_string = "\"{}\":\"{}\"".format("agno", len(relevant_results_queryset.filter(shuffle="shuffles")))
    dist_string = "\"{}\":\"{}\"".format("dist", len(relevant_results_queryset.filter(shuffle="distshuffles")))
    edge_string = "\"{}\":\"{}\"".format("edge", len(relevant_results_queryset.filter(shuffle="edgeshuffles")))
    return HttpResponse(
        "{\n    " + ",\n    ".join(
            [sign_string, none_string, agno_string, dist_string, edge_string, ]
        ) + "\n}"
    )


def rest_refresh(request, job_name):
    """ Execute the celery task to refresh the result list, and return json with the task_id necessary for
        checking on the job periodically and updating the progress bar.
    """

    jobs_id = celery_id_from_name(job_name)
    plots_in_progress = celery_plots_in_progress()

    # print("  in rest_refresh, job is {}, id is {}.".format(job_name, jobs_id))

    if job_name in ["collect_jobs", "update_jobs", ]:
        if jobs_id is None:
            print("NEW: rest_refresh got job '{}', no id returned. Re-building results database.".format(job_name))
            if job_name == "collect_jobs":
                clear_jobs()
                celery_result = collect_jobs.delay("/data", rebuild=True)
            else:
                celery_result = collect_jobs.delay("/data", rebuild=False)
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "traintest" in job_name:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. Building new plot.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = build_plot.delay(job_name[:8].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "performance" in job_name:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New performance assessment.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = assess_performance.delay(job_name[:8].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    else:
        print("I don't understand job_name '{}'".format(job_name))

    return HttpResponse("{" + "\"task_id\": \"{}\"".format(jobs_id) + "}")


def rest_latest(request):
    """ Return json with the latest state of the data. """

    r = ResultSummary.empty() if ResultSummary.objects.count() == 0 else ResultSummary.objects.latest('summary_date')
    return HttpResponse(r.to_json())

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