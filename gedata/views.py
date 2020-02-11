from __future__ import absolute_import, unicode_literals

import os

from django.http import HttpResponse
from django.shortcuts import render
from django.views import generic

from ge_data_manager.celery import celery_id_from_name, celery_plots_in_progress

from .tasks import clear_jobs, collect_jobs, interpret_descriptor, comp_from_signature
from .tasks import assess_mantel, assess_overlap, assess_performance, assess_everything
from .tasks import clear_macro_caches, clear_micro_caches
from .models import PushResult, ResultSummary
from .forms import thresholds


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

class InventoryView(generic.ListView):
    model = PushResult
    template_name = 'gedata/inventory.html'

    def get_context_data(self, **kwargs):
        base_str_n = '<p>{n_none:,} ({n_agno:,}+{n_dist:,}+{n_edge:,}+{n_be04:,}+{n_be08:,}+{n_be16:,}) '

        def span_str(r_id, metric, ext, icon, threshold=""):
            """ Return the html <span ...><a href=...><i ...></i></a></span> for a particular item. """
            if metric == "performance":
                img_file = '{rid}_{metric}.{ext}'.format(rid=r_id, metric=metric, ext=ext)
            else:
                img_file = '{rid}{threshold}_{metric}.{ext}'.format(
                    rid=r_id, threshold=threshold, metric=metric, ext=ext,
                )

            img_url = '/static/gedata/plots/{img_file}'.format(img_file=img_file)
            if os.path.isfile('/data/plots/{img_file}'.format(img_file=img_file)):
                anchor = '<a href="{url}" target="_blank"><i class="fas {icon}"></i></a>'.format(
                    url=img_url, icon=icon
                )
            else:
                anchor = '<i class="fal {icon}"></i>'.format(
                    icon=icon
                )
            return '<span id="{rid}{metric}">{anchor}</span>'.format(
                rid=rid, metric=metric, anchor=anchor
            )

        initial_queryset = PushResult.objects.filter(
            samp="glasser", prob="fornito", algo='smrt', batch__startswith='train',
        )

        print("Building inventory from {:,} initial results.".format(len(initial_queryset)))

        context = super(InventoryView, self).get_context_data(**kwargs)
        context['masks'] = ["00", "16", "32", "64"]
        for p in ['w', 'g']:
            for s in ['w', 'g']:
                for c in ['hcp', 'nki', 'f__', 'n__', 'fn_']:
                    psc_id = "{}{}{}{}".format(c, p, s, 's')
                    context[psc_id] = {}
                    for m in ['00', '16', '32', '64']:
                        for nrm in ['s', '_']:
                            for xv in ['2', '4', ]:
                                min_split = 0
                                max_split = 0
                                if xv == '2':
                                    min_split = 200
                                    max_split = 299
                                elif xv == '4':
                                    min_split = 400
                                    max_split = 499
                                final_queryset = initial_queryset.filter(
                                    comp=comp_from_signature(c + p),
                                    parby="glasser" if p == "g" else "wellid",
                                    splby="glasser" if s == "g" else "wellid",
                                    mask='none' if m == "00" else m,
                                    norm='srs' if nrm == 's' else 'none',
                                    split__gte=min_split, split__lte=max_split
                                )
                                rid = "{}{}{}{}{}{}{}".format(c, p, s, m, 's', nrm, xv)
                                if len(final_queryset) > 0:
                                    print("  {:,} are for {}".format(len(final_queryset), rid))

                                span_strings = "<br />".join([" ".join([
                                    span_str(rid, "mantel", "png", "fa-box-up", threshold[0]),
                                    span_str(rid, "overlap", "png", "fa-object-group", threshold[0]),
                                    span_str(rid, "genes", "html", "fa-dna", threshold[0]),
                                    span_str(rid, "ranked", "csv", "fa-list-ol", threshold[0]),
                                    "@", threshold[1],
                                    span_str(rid, "report", "html", "fa-file-chart-line", 'peak') if threshold[0] == 'peak' else '',
                                ]) for threshold in thresholds])
                                buttons = " ".join([
                                    "<span id=\"inventory_string\" style=\"display: none;\"></span>",
                                    "<button class=\"btn\" onclick=\"{}\">{}</button>".format(
                                        "assessEverything('image_{}', 'inventory_string', '{}');".format(rid, rid),
                                        "<i class='fas fa-abacus'></i>",
                                    ),
                                    "<button class=\"btn\" onclick=\"{}\">{}</button>".format(
                                        "removeEverything('image_{}', '{}');".format(rid, rid),
                                        "<i class='fas fa-trash'></i>",
                                    ),
                                    "<div id=\"image_{}\"><img src=\"\"></div>".format(rid),
                                ])
                                context[rid] = " ".join([
                                    base_str_n.format(
                                        n_none=len(final_queryset.filter(shuffle="derivatives")),
                                        n_agno=len(final_queryset.filter(shuffle="shuffles")),
                                        n_dist=len(final_queryset.filter(shuffle="distshuffles")),
                                        n_edge=len(final_queryset.filter(shuffle="edgeshuffles")),
                                        n_be04=len(final_queryset.filter(shuffle="edge04shuffles")),
                                        n_be08=len(final_queryset.filter(shuffle="edge08shuffles")),
                                        n_be16=len(final_queryset.filter(shuffle="edge16shuffles")),
                                    ),
                                    span_str(rid, "performance", "png", "fa-chart-line"), "<br />",
                                    span_strings, "<br />",
                                    "<div style=\"text-align: right;\">", buttons, "</div>"
                                ])

                                # Duplicate the data so it can be looked up two different ways.
                                context[psc_id][m] = context[rid]
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
    be04_string = "\"{}\":\"{}\"".format("be04", len(relevant_results_queryset.filter(shuffle="edge04shuffles")))
    be08_string = "\"{}\":\"{}\"".format("be08", len(relevant_results_queryset.filter(shuffle="edge08shuffles")))
    be16_string = "\"{}\":\"{}\"".format("be16", len(relevant_results_queryset.filter(shuffle="edge16shuffles")))
    return HttpResponse(
        "{\n    " + ",\n    ".join(
            [sign_string, none_string, agno_string, dist_string, edge_string, be04_string, be08_string, be16_string, ]
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
    elif "mantel" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New Mantel assessment.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = assess_mantel.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "everything" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New assessment of everything.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = assess_everything.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "performance" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New performance assessment.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = assess_performance.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "overlap" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print(
                "NEW: rest_refresh got job '{}', no id returned. New overlap assessment.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = assess_overlap.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "clearmacro" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New assessment of everything.".format(job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = clear_macro_caches.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    elif "clearmicro" in job_name.rsplit('_', 1)[1]:
        if job_name in plots_in_progress:
            print("DUPE: {} requested, but is already being worked on.".format(job_name))
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New assessment of everything.".format(
                job_name))
            for plot in plots_in_progress:
                print("     already building {}".format(plot))
            celery_result = clear_micro_caches.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            jobs_id = celery_result.task_id
            print("     new id for '{}' is '{}'.".format(job_name, jobs_id))
    else:
        print("I don't understand job_name '{}'".format(job_name))

    return HttpResponse("{" + "\"task_id\": \"{}\"".format(jobs_id) + "}")


def rest_latest(request):
    """ Return json with the latest state of the data. """

    r = ResultSummary.empty() if ResultSummary.objects.count() == 0 else ResultSummary.objects.latest('summary_date')
    return HttpResponse(r.to_json())


def inventory(request):
    """ Render inventory """

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