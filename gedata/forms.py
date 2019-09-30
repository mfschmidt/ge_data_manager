from django import forms
from django.shortcuts import render
from django.db.utils import ProgrammingError

from os import path
from .models import PushResult, ResultSummary


def unique_tuples(k, enumerated=False):
    """ Return a list of tuples, containing all values matching the k key, useful for select box widgets """

    # Upon initial run, the database may not exist yet and trigger a ProgrammingError when queried.
    unique_values = [] if PushResult.objects.count() == 0 else PushResult.objects.order_by().values_list(k).distinct()

    vs = (('*', '*', ), )
    for i, v in enumerate(sorted(unique_values)):
        if enumerated:
            vs += ((str(i), ) + v, )
        else:
            vs += (v + v, )
    return vs

class FilterForm(forms.Form):
    """ Focus on split halves for now, ignoring less interesting variables. """

    # subject = forms.ChoiceField(
    #     label="subject (sub)", widget=forms.Select, choices=unique_tuples('subject'), initial='*')
    # hemisphere = forms.ChoiceField(
    #     label="hemisphere (hem)", widget=forms.Select, choices=unique_tuples('hemisphere'), initial='*')
    # cortex = forms.ChoiceField(
    #     label="cortex (ctx)", widget=forms.Select, choices=unique_tuples('cortex'), initial='*')
    # probes = forms.ChoiceField(
    #     label="probes (prb)", widget=forms.Select, choices=unique_tuples('probes'), initial='*')
    # target = forms.ChoiceField(
    #     label="target (tgt)", widget=forms.Select, choices=unique_tuples('target'), initial='*')
    parby = forms.ChoiceField(
        label="parcel by (parby)", widget=forms.Select, choices=unique_tuples('parby'), initial='glasser')
    splby = forms.ChoiceField(
        label="split by (splby)", widget=forms.Select, choices=unique_tuples('splby'), initial='wellid')
    algo = forms.ChoiceField(
        label="algorithm (algo)", widget=forms.Select, choices=unique_tuples('algo'), initial='*')
    # normalization = forms.ChoiceField(
    #     label="normalization (norm)", widget=forms.Select, choices=unique_tuples('normalization'), initial='*')
    comp = forms.ChoiceField(
        label="comparator (comp)", widget=forms.Select, choices=unique_tuples('comp'), initial='*')
    mask = forms.ChoiceField(
        label="mask (mask)", widget=forms.Select, choices=unique_tuples('mask'), initial='none')
    # adjustment = forms.ChoiceField(
    #     label="adjustment (adj)", widget=forms.Select, choices=unique_tuples('adjustment'), initial='*')

def filter_results(request):
    """ Render the FilterForm. """

    submitted = False
    query_set = PushResult.objects.none()
    comments = []
    result_expression = ""
    result_split = ""
    result_process = ""
    result_file = ""

    if request.method == 'POST':
        # Data were POSTed, so we can populate the fields with them.
        form = FilterForm(request.POST)
        if form.is_valid():
            # Use the valid form data to filter results and display them.
            cd = form.cleaned_data
            result_expression = "sub-all_hem-A_samp-glasser_prob-fornito"
            result_split = "parby-{parby}_splby-{splby}_batch-*".format(**cd)
            result_process = "tgt-*_algo-{algo}".format(**cd)
            result_file = "sub-all_comp-{comp}_mask-{mask}_norm-none_adj-none.tsv".format(**cd)
            comments.append("parcel by {}, split by {}, algo='{}', comp='{}', mask='{}'".format(
                cd['parby'], cd['splby'], cd['algo'], cd['comp'], cd['mask']
            ))
            query_set = PushResult.objects.all()  # filter(shuffle='derivatives')
            for filter_term in ['parby', 'splby', 'algo', 'comp', 'mask']:
                if cd[filter_term] != '*':
                    query_set = query_set.filter(
                        sub='all', samp='glasser', prob='fornito', tgt='max', **{filter_term: cd[filter_term]}
                    )
                    comments.append("filtered {} by {}, {:,} remain (+{:,}+{:,}+{:,} shuffles)".format(
                        filter_term, cd[filter_term],
                        len(query_set.filter(shuffle='derivatives')),
                        len(query_set.filter(shuffle='shuffles')),
                        len(query_set.filter(shuffle='distshuffles')),
                        len(query_set.filter(shuffle='edgeshuffles'))
                    ))
    else:
        # Create a blank form
        form = FilterForm()
        if 'submitted' in request.GET:
            submitted = True

    print("Rendering filter, final queryset is {:,} results long.".format(len(query_set)))

    return render(request, 'gedata/filter.html', {
        'form': form,
        'submitted': submitted,
        'pushresult_list': query_set.filter(shuffle='derivatives'),
        'result_path': path.join("derivatives", result_expression, result_split, result_process, result_file),
        'comments': comments,
        # 'latest_result_summary': ResultSummary.objects.latest('summary_date')
    })


class CompareForm(forms.Form):
    complete_sets = [
        ('hcpgg00s', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-none'),
        ('hcpgg16s', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-16'),
        ('hcpgg32s', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-32'),
        ('hcpgg64s', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-64'),
        ('hcpgw00s', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-none'),
        ('hcpgw16s', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-16'),
        ('hcpgw32s', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-32'),
        ('hcpgw64s', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-64'),
        ('hcpww00s', 'parby-wellid_splby-wellid ~ hcpniftismoothgrandmeansim mask-none'),
        ('nkigg00s', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-none'),
        ('nkigg16s', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-16'),
        ('nkigg32s', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-32'),
        ('nkigg64s', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-64'),
        ('nkigw00s', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-none'),
        ('nkigw16s', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-16'),
        ('nkigw32s', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-32'),
        ('nkigw64s', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-64'),
        ('nkiwg00s', 'parby-wellid_splby-glasser ~ indiconnsim mask-none'),
        ('nkiww00s', 'parby-wellid_splby-wellid ~ indiconnsim mask-none'),
    ]
    left_set = forms.ChoiceField(
        label="start with", widget=forms.Select, choices=complete_sets, initial='hcpgg00sc',
    )
    right_set = forms.ChoiceField(
        label="compare with", widget=forms.Select, choices=complete_sets, initial='nkigg00sc',
    )


def image_dict_from_selection(selection):
    """ Convert a selection ID into a dictionary with image data for displaying the correct plot. """

    image_dict = {
        'url': "/static/plots/train_test_{}.png".format(selection.lower()),
    }

    if selection[:3].lower() == "hcp":
        if selection[3].lower() == "g":
            image_dict['comp'] = "glasserconnectivitysim"
        elif selection[3].lower() == "w":
            image_dict['comp'] = "hcpniftismoothgrandmeansim"
    elif selection[:3].lower() == "nki":
        if selection[3].lower() == "g":
            image_dict['comp'] = "indiglasserconnsim"
        elif selection[3].lower() == "w":
            image_dict['comp'] = "indiconnsim"

    image_dict['description'] = "parby-{}_splby-{} ~ {} mask-{} {}".format(
        "glasser" if selection[3].lower() == "g" else "wellid",
        "glasser" if selection[4].lower() == "g" else "wellid",
        image_dict['comp'],
        "none" if selection[5:] == "00" else selection[5:],
        "once" if selection[7] == "o" else "smart",
    )

    return image_dict


def image_dict_from_selections(clean_form_data, side):
    """ Convert a collection of selections into a single string allowing selection of the correct plot image. """

    prefix='na'
    summary_string = "empty"
    summary_template = "{comp}{pby}{sby}{mask}{algo}{test_with_mask}"
    if side.upper()[0] == "L":
        prefix = 'train_test'
        summary_string = summary_template.format(
            comp=clean_form_data['left_comp'].lower(),
            pby=clean_form_data['left_parcel'].lower()[0],
            sby=clean_form_data['left_split'].lower()[0],
            mask=clean_form_data['left_train_mask'],
            algo=clean_form_data['left_algo'],
        )
    elif side.upper()[0] == "R":
        prefix = 'train_test'
        summary_string = summary_template.format(
            comp=clean_form_data['right_comp'].lower(),
            pby=clean_form_data['right_parcel'].lower()[0],
            sby=clean_form_data['right_split'].lower()[0],
            mask=clean_form_data['right_train_mask'],
            algo=clean_form_data['right_algo'],
        )
    elif side.lower() == "performance":
        prefix='performance'
        summary_string = summary_template.format(
            comp=clean_form_data['comp'].lower(),
            pby=clean_form_data['parcel'].lower()[0],
            sby=clean_form_data['split'].lower()[0],
            mask=clean_form_data['train_mask'],
            algo=clean_form_data['algo'],
        )
    image_dict = {
        'url': "/static/plots/{}_{}.png".format(prefix, summary_string),
        'alt': summary_string,
    }
    return image_dict


def compare_results(request):
    """ Render the CompareForm """

    submitted = False
    left_image = {'url': '/static/gedata/empty.png', 'description': 'nonexistent'}
    right_image = {'url': '/static/gedata/empty.png', 'description': 'nonexistent'}

    if request.method == 'POST':
        form = CompareForm(request.POST)
        if form.is_valid():
            left_image = image_dict_from_selection(form.cleaned_data['left_set'])
            right_image = image_dict_from_selection(form.cleaned_data['right_set'])
    else:
        # Create a blank form
        form = CompareForm()
        if 'submitted' in request.GET:
            submitted = True

    return render(request, 'gedata/compare.html', {
        'form': form,
        'submitted': submitted,
        'left_image': left_image,
        'right_image': right_image,
        # 'latest_result_summary': ResultSummary.objects.latest('summary_date'),
    })


class ComparisonForm(forms.Form):
    parcels = [ ('w', 'wellid'), ('g', 'Glasser'), ]
    comps = [ ('nki', 'NKI'), ('hcp', 'HCP'), ]
    masks = [ ('00', 'none'), ('16', '16'), ('32', '32'), ('64', '64'), ]
    algos = [ ('s', 'smrt'), ('o', 'once'), ]

    left_parcel = forms.ChoiceField(label="parcel", widget=forms.Select, choices=parcels, initial="w")
    left_split = forms.ChoiceField(label="split by", widget=forms.Select, choices=parcels, initial="w")
    left_comp = forms.ChoiceField(label="connectivity", widget=forms.Select, choices=comps, initial="nki")
    left_train_mask = forms.ChoiceField(label="train mask", widget=forms.Select, choices=masks, initial="00")
    left_algo = forms.ChoiceField(label="algorithm", widget=forms.Select, choices=algos, initial="s")

    right_parcel = forms.ChoiceField(label="parcel", widget=forms.Select, choices=parcels, initial="w")
    right_split = forms.ChoiceField(label="split by", widget=forms.Select, choices=parcels, initial="w")
    right_comp = forms.ChoiceField(label="connectivity", widget=forms.Select, choices=comps, initial="nki")
    right_train_mask = forms.ChoiceField(label="train mask", widget=forms.Select, choices=masks, initial="00")
    right_algo = forms.ChoiceField(label="algorithm", widget=forms.Select, choices=algos, initial="s")


def comparison_results(request):
    """ Render the ComparisonForm """

    submitted = False
    left_image = {'url': '/static/gedata/empty.png', 'description': 'nonexistent'}
    right_image = {'url': '/static/gedata/empty.png', 'description': 'nonexistent'}

    if request.method == 'POST':
        form = ComparisonForm(request.POST)
        if form.is_valid():
            left_image = image_dict_from_selections(form.cleaned_data, 'left')
            right_image = image_dict_from_selections(form.cleaned_data, 'right')
            print("We calculated the image names, {} & {}, in python. I didn't think we'd ever POST form data.".format(
                left_image, right_image
            ))
    else:
        # Create a blank form
        form = ComparisonForm()
        if 'submitted' in request.GET:
            submitted = True

    return render(request, 'gedata/comparison.html', {
        'form': form,
        'submitted': submitted,
        'left_image': left_image,
        'right_image': right_image,
        # 'latest_result_summary': ResultSummary.objects.latest('summary_date'),
    })


class PerformanceForm(forms.Form):
    parcels = [ ('w', 'wellid'), ('g', 'Glasser'), ]
    comps = [ ('nki', 'NKI'), ('hcp', 'HCP'), ]
    masks = [ ('00', 'none'), ('16', '16'), ('32', '32'), ('64', '64'), ]
    algos = [ ('s', 'smrt'), ('o', 'once'), ]

    parcel = forms.ChoiceField(label="parcel", widget=forms.Select, choices=parcels, initial="w")
    split = forms.ChoiceField(label="split by", widget=forms.Select, choices=parcels, initial="w")
    comp = forms.ChoiceField(label="connectivity", widget=forms.Select, choices=comps, initial="nki")
    train_mask = forms.ChoiceField(label="train mask", widget=forms.Select, choices=masks, initial="00")
    algo = forms.ChoiceField(label="algorithm", widget=forms.Select, choices=algos, initial="s")


def performance(request):
    """ Render the ComparisonForm """

    submitted = False
    image = {'url': '/static/gedata/empty.png', 'description': 'nonexistent'}

    if request.method == 'POST':
        form = ComparisonForm(request.POST)
        if form.is_valid():
            image = image_dict_from_selections(form.cleaned_data, 'performance')
            print("We calculated the image name, {}, in python. I didn't think we'd ever POST form data.".format(
                image
            ))
    else:
        # Create a blank form
        form = PerformanceForm()
        if 'submitted' in request.GET:
            submitted = True

    return render(request, 'gedata/performance.html', {
        'form': form,
        'submitted': submitted,
        'image': image,
        # 'latest_result_summary': ResultSummary.objects.latest('summary_date'),
    })

