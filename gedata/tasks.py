from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery_progress.backend import ProgressRecorder
import time
from datetime import datetime
from django.utils import timezone

import os
import re


from .models import PushResult, ResultSummary


@shared_task
def add(x, y):
    return x + y


@shared_task(bind=True)
def test_task(self, seconds):
    """ A test task to run for a given number of seconds

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    :param seconds: How many seconds to run
    """

    progress_recorder = ProgressRecorder(self)
    for i in range(seconds):
        time.sleep(1)
        progress_recorder.set_progress(i + 1, seconds)
    return 'done'


@shared_task
def clear_jobs():
    PushResult.objects.all().delete()


@shared_task(bind=True)
def collect_jobs(self, data_path="/data", new_only=False):
    """ Traverse the output and populate the database with completed results.

    :param self: available through "bind=True", allows a reference to the celery task this function becomes a part of.
    :param data_path: default /data, base path to all of the results
    :param new_only: default False, If true, check each file found and only add new ones.
    """

    progress_recorder = ProgressRecorder(self)

    def json_contents(json_file):
        """ Parse contents of json file into a dict

            I tried the standard json parser, but had repeated issues and failures.
            Regex works well and the code is still fairly clean.
        """
        items = {}
        with open(json_file, "r") as jf:
            for line in jf.readlines():
                clean_line = line.strip().rstrip(",").replace(": ", ":")
                m = re.match(".*\"(?P<k>.+)\":\"(?P<v>.+)\".*", clean_line)
                if m:
                    k = m.group('k')
                    v = m.group('v')
                    items[k] = v
        return items


    def seconds_elapsed(elapsed):
        parts = elapsed.split(":")
        if len(parts) != 3:
            return 0
        seconds = int(float(parts[2]))
        minutes = int(parts[1])
        if "days" in parts[0]:
            hours = int(parts[0].split(" days, ")[1])
            days = int(parts[0].split(" days, ")[0])
        elif "day" in parts[0]:
            hours = int(parts[0].split(" day, ")[1])
            days = int(parts[0].split(" day, ")[0])
        else:
            hours = int(parts[0])
            days = 0
        return seconds + (minutes * 60) + (hours * 3600) + (days * 3600 * 24)

    results = []
    i = 0
    n = 0

    # Get a list of files to parse.
    for shuffle in ['derivatives', 'shuffles', 'edgeshuffles', 'distshuffles', ]:
        d = os.path.join(data_path, shuffle)
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d, topdown=True):
                for f in files:
                    if f[(-4):] == "json":
                        result = {
                            'data_dir': data_path,
                            'shuffle': shuffle,
                            'path': root,
                            'json_file': f,
                            'tsv_file': f.replace("json", "tsv"),
                            'log_file': f.replace("json", "log"),
                        }
                        results.append(result)
                        n += 1
                        progress_recorder.set_progress(i, n) #  Calculating...
                        # self.update_state(state="PROGRESS",
                        #                   meta={'current': i, 'total': i, 'message': "Discovering file {:,}".format(i)}
                        # )

    # Gather information about each result.
    for i, result in enumerate(results):
        # First, parse out the BIDS key-value pairs from the path.
        bids_pairs = []
        bids_dict = {'root': result['path'], 'name': result['json_file']}
        fs_parts = os.path.join(result['path'], result['json_file'])[: -5 :].split(os.sep)
        for fs_part in fs_parts:
            if '-' in fs_part:
                pairs = fs_part.split("_")
                for pair in pairs:
                    if '-' in pair:
                        p = pair.split("-")
                        bids_pairs.append((p[0], p[1]))
                    else:
                        # There should never be an 'extra' but we catch it to debug problems.
                        bids_pairs.append(('extra', pair))
        for bp in bids_pairs:
            bids_dict[bp[0]] = bp[1]
        result.update(bids_dict)

        # Second, parse the key-value pairs from the json file containing processing information.
        json_dict = json_contents(os.path.join(result['path'], result['json_file']))
        result.update(json_dict)

        # Finally, put it all into a model for storage in the database.
        r = PushResult(
            json_path=os.path.join(result['path'], result['json_file']),
            tsv_path=os.path.join(result['path'], result['json_file'].replace("json", "tsv")),
            log_path=os.path.join(result['path'], result['json_file'].replace("json", "log")),
            # For dates, we assume EDT (-0400) as most runs were in EDT. For those in EST, the hour makes no real difference.
            start_date=datetime.strptime(result.get("began", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"),
            end_date=datetime.strptime(result.get("completed", "1900-01-01 00:00:00") + "-0400", "%Y-%m-%d %H:%M:%S%z"),
            host=result.get("host", ""),
            command=result.get("command", ""),
            version=result.get("pygest version", ""),
            duration=seconds_elapsed(result.get("elapsed", "")),
            shuffle=result.get("shuffle", ""),
            subject=result.get("sub", ""),
            hemisphere=result.get("hem", " ").upper()[0],
            cortex=result.get("ctx", ""),
            probes=result.get("prb", ""),
            target=result.get("tgt", ""),
            algorithm=result.get("alg", ""),
            normalization=result.get("norm", ""),
            comparator=result.get("cmp", ""),
            mask=result.get("msk", ""),
            adjustment=result.get("adj", ""),
            seed=int(result.get("seed", "0")),
            columns=0,
            rows=0,
        )
        r.save()
        progress_recorder.set_progress(i, n)

    if PushResult.objects.count() > 0:
        s = ResultSummary(
            summary_date=timezone.now(),
            num_results=PushResult.objects.count(),
            num_actuals=PushResult.objects.filter(shuffle='derivatives').count(),
            num_shuffles=PushResult.objects.filter(shuffle='shuffles').count(),
            num_distshuffles=PushResult.objects.filter(shuffle='distshuffles').count(),
            num_edgeshuffles=PushResult.objects.filter(shuffle='edgeshuffles').count(),
            num_splits=0,
        )
        s.save()

    # results_df = pd.DataFrame(results)
    # return results_df
