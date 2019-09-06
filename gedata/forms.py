from django import forms
from django.shortcuts import render

from os import path
from .models import PushResult, ResultSummary


def unique_tuples(k, enumerated=False):
    """ Return a list of tuples, containing all values matching the k key, useful for select box widgets """

    unique_values = PushResult.objects.order_by().values_list(k).distinct()
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
        'latest_result_summary': ResultSummary.objects.latest('summary_date')
    })


class CompareForm(forms.Form):
    complete_sets = [
        ('HCPGG00', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-none'),
        ('HCPGG16', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-16'),
        ('HCPGG32', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-32'),
        ('HCPGG64', 'parby-glasser_splby-glasser ~ glasserconnectivitysim mask-64'),
        ('HCPGW00', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-none'),
        ('HCPGW16', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-16'),
        ('HCPGW32', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-32'),
        ('HCPGW64', 'parby-glasser_splby-wellid ~ glasserconnectivitysim mask-64'),
        ('HCPWW00', 'parby-wellid_splby-wellid ~ hcpniftismoothgrandmeansim mask-none'),
        ('NKIGG00', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-none'),
        ('NKIGG16', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-16'),
        ('NKIGG32', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-32'),
        ('NKIGG64', 'parby-glasser_splby-glasser ~ indiglasserconnsim mask-64'),
        ('NKIGW00', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-none'),
        ('NKIGW16', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-16'),
        ('NKIGW32', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-32'),
        ('NKIGW64', 'parby-glasser_splby-wellid ~ indiglasserconnsim mask-64'),
        ('NKIWG00', 'parby-wellid_splby-glasser ~ indiconnsim mask-none'),
        ('NKIWW00', 'parby-wellid_splby-wellid ~ indiconnsim mask-none'),
    ]
    left_set = forms.ChoiceField(
        label="start with", widget=forms.Select, choices=complete_sets, initial='HCPGG00',
    )
    right_set = forms.ChoiceField(
        label="compare with", widget=forms.Select, choices=complete_sets, initial='NKIGG00',
    )


def image_dict_from_selection(selection):
    """ Convert a selection ID into a dictionary with image data for displaying the correct plot. """

    image_dict = {
        'url': "/static/plots/train_test_{}.png".format(selection.lower()),
    }

    if selection[:3].upper() == "HCP":
        if selection[3].upper() == "G":
            image_dict['comp'] = "glasserconnectivitysim"
        elif selection[3].upper() == "W":
            image_dict['comp'] = "hcpniftismoothgrandmeansim"
    elif selection[:3].upper() == "NKI":
        if selection[3].upper() == "G":
            image_dict['comp'] = "indiglasserconnsim"
        elif selection[3].upper() == "W":
            image_dict['comp'] = "indiconnsim"

    image_dict['description'] = "parby-{}_splby-{} ~ {} mask-{}".format(
        "glasser" if selection[3].upper() == "G" else "wellid",
        "glasser" if selection[4].upper() == "G" else "wellid",
        image_dict['comp'],
        "none" if selection[5:] == "00" else selection[5:],
    )

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
        'latest_result_summary': ResultSummary.objects.latest('summary_date'),
    })
