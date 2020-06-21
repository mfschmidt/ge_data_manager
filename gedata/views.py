from __future__ import absolute_import, unicode_literals

import os

from django.http import HttpResponse
from django.shortcuts import render
from django.views import generic

from ge_data_manager.celery import celery_id_from_name, celery_tasks_in_progress

from .tasks import clear_all_jobs, interpret_descriptor, gather_results
from .tasks import assess_mantel, assess_overlap, assess_performance, assess_everything, just_genes
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
        return PushResult.objects.filter(shuf='none')


class NewInventoryView(generic.ListView):
    model = PushResult
    template_name = "gedata/newinventory.html"

    # Order by reverse comp (hcp, then fn), reverse pby&sby (ww, gg), reverse split (4*, then 2*)
    # Just coincidentally, -shuf works for ('none', 'dist', 'be04', 'agno') but may break later and require more code.
    ordering = ['-comp', 'resample', '-splby', 'mask', '-shuf', 'split', 'seed']

    # Returns self.object_list to the template.


class InventoryView(generic.ListView):
    model = PushResult
    template_name = 'gedata/inventory.html'

    # Override django view function, return context for display
    def get_context_data(self, **kwargs):
        base_str_n = "{n_none:,} real, {n_shuf:,} permutations"

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
                                rid = "{}{}{}{}{}{}{}".format("image_c", p, s, m, 's', nrm, xv)
                                final_queryset = initial_queryset.filter(
                                    descriptor=rid,
                                )
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
                                        "assessEverything('{}');".format(rid),
                                        "<i class='fas fa-abacus'></i>",
                                    ),
                                    "<button class=\"btn\" onclick=\"{}\">{}</button>".format(
                                        "assessJustGenes('{}');".format(rid),
                                        "<i class='fas fa-dna'></i>",
                                    ),
                                    "<button class=\"btn\" onclick=\"{}\">{}</button>".format(
                                        "removeEverything('{}');".format(rid, rid),
                                        "<i class='fas fa-trash'></i>",
                                    ),
                                    "<div id=\"spinner_{}\"><span class=\"leave_blank\"></span></div>".format(rid),
                                ])
                                n_none = len(final_queryset.filter(shuf="none"))
                                context[rid] = " ".join([
                                    base_str_n.format(n_none=n_none, n_shuf=len(final_queryset) - n_none,),
                                    span_str(rid, "performance", "png", "fa-chart-line"), "<br />",
                                    span_strings, "<br />",
                                    "<div style=\"text-align: right;\">", buttons, "</div>"
                                ])

                                # Duplicate the data so it can be looked up two different ways.
                                context[psc_id][m] = context[rid]
        return context

    def get_queryset(self):
        return PushResult.objects.filter(shuf='none')


def rest_inventory(request, signature):
    """ From an inventory id, like 'hcpww00s', return four-part inventory json. """

    comp, parby, splby, mask, algo, phase, opposite_phase, relevant_results_queryset = interpret_descriptor(signature)
    sign_string = "\"{}\":\"{}\"".format("signature", signature)
    none_string = "\"{}\":\"{}\"".format("none", len(relevant_results_queryset.filter(shuf="none")))
    agno_string = "\"{}\":\"{}\"".format("agno", len(relevant_results_queryset.filter(shuf="agno")))
    dist_string = "\"{}\":\"{}\"".format("dist", len(relevant_results_queryset.filter(shuf="dist")))
    edge_string = "\"{}\":\"{}\"".format("edge", len(relevant_results_queryset.filter(shuf="edge")))
    be04_string = "\"{}\":\"{}\"".format("be04", len(relevant_results_queryset.filter(shuf="be04")))
    be08_string = "\"{}\":\"{}\"".format("be08", len(relevant_results_queryset.filter(shuf="be08")))
    be16_string = "\"{}\":\"{}\"".format("be16", len(relevant_results_queryset.filter(shuf="be16")))
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
    tasks_in_progress = celery_tasks_in_progress()
    task_handle = job_name.rsplit("_", 1)[1]

    print("  in rest_refresh, job is {}, id is {}.".format(job_name, jobs_id))

    def handle_task(fn):
        if task_handle in tasks_in_progress:
            print("DUPE: (job_name: {}, task_handle: {}) requested, but is already being worked on.".format(
                job_name, task_handle
            ))
            return jobs_id
        else:
            print("NEW: rest_refresh got job '{}', no id returned. New assessment.".format(job_name))
            for plot in tasks_in_progress:
                print("     already building {}".format(plot))
            celery_result = fn.delay(job_name.rsplit('_', 1)[0].lower(), data_root="/data")
            print("     new id for '{}' is '{}'.".format(job_name, celery_result.task_id))
            return celery_result.task_id


    if job_name == "global_refresh":
        if jobs_id is None:
            print("NEW: Starting a refresh of all data")
            celery_result = gather_results.delay(data_root="/data")
            print("     Submitted gather and populate of results (id={}).".format(celery_result.task_id))
            jobs_id = celery_result.task_id
    elif job_name == "global_clear":
        if jobs_id is None:
            print("NEW: Starting a clear of all data")
            celery_result = clear_all_jobs.delay()
            print("     Submitted result clear-out (id={}).".format(celery_result.task_id))
            jobs_id = celery_result.task_id
    elif task_handle == "justgenes":
        jobs_id = handle_task(just_genes)
    elif task_handle == "mantel":
        jobs_id = handle_task(assess_mantel)
    elif task_handle == "everything":
        jobs_id = handle_task(assess_everything)
    elif task_handle == "performance":
        jobs_id = handle_task(assess_performance)
    elif task_handle == "overlap":
        jobs_id = handle_task(assess_overlap)
    elif task_handle == "clearmacro":
        jobs_id = handle_task(clear_macro_caches)
    elif task_handle == "clearmicro":
        jobs_id = handle_task(clear_micro_caches)
    else:
        print("I don't understand job_name '{}'".format(job_name))

    return HttpResponse("{" + "\"task_id\": \"{}\"".format(jobs_id) + "}")


def rest_latest(request):
    """ Return json with the latest state of the data. """

    r = ResultSummary.empty() if ResultSummary.objects.count() == 0 else ResultSummary.objects.latest('summary_date')
    print("Asked about the latest status (of {}): ".format(ResultSummary.objects.count()), r.to_json())
    return HttpResponse(r.to_json())
