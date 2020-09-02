from __future__ import absolute_import, unicode_literals

from django.db import models
from django.utils import timezone as django_timezone
import datetime


# Create your models here.
class PushResult(models.Model):
    """ Each run of PyGEST has many features. They are recorded in the PushResult class. """

    # Index for rapid searching of inventory
    sourcedata = models.CharField(db_index=True, max_length=128, default="")
    descriptor = models.CharField(db_index=True, max_length=16, default="")

    # File information
    json_path = models.FilePathField(path="/data", default="/data", max_length=256)
    json_present = models.BooleanField(default=False)
    tsv_path = models.FilePathField(path="/data", default="/data", max_length=256)
    tsv_present = models.BooleanField(default=False)
    log_path = models.FilePathField(path="/data", default="/data", max_length=256)
    log_present = models.BooleanField(default=False)
    summary_path = models.FilePathField(path="/data", default="/data", max_length=256)
    summary_present = models.BooleanField(default=False)
    entrez_path = models.FilePathField(path="/data", default="/data", max_length=256)
    entrez_present = models.BooleanField(default=False)
    ejgo_path = models.FilePathField(path="/data", default="/data", max_length=256)
    ejgo_present = models.BooleanField(default=False)

    # Execution details
    start_date = models.DateTimeField('date_started')
    end_date = models.DateTimeField('date_completed')
    host = models.CharField(max_length=16)
    command = models.CharField(max_length=384)
    version = models.CharField(max_length=64)  # 'running uninstalled without version' longest
    # duration is in seconds
    duration = models.IntegerField()

    # Data preparation and execution details
    shuf = models.CharField(db_index=True, max_length=16)  # 'none', 'agno', 'be04', 'dist', etc.
    resample = models.CharField(max_length=16)  # 'whole', 'split-half', 'split-quarter'
    sub = models.CharField(max_length=32)
    hem = models.CharField(max_length=1)
    samp = models.CharField(max_length=16)
    prob = models.CharField(max_length=16)
    parby = models.CharField(max_length=16)
    splby = models.CharField(max_length=16)
    batch = models.CharField(max_length=16)
    tgt = models.CharField(max_length=3)
    algo = models.CharField(max_length=8)
    norm = models.CharField(max_length=16)
    comp = models.CharField(max_length=48)
    mask = models.CharField(max_length=32)
    adj = models.CharField(max_length=16)
    split = models.IntegerField(default=0)
    seed = models.IntegerField(default=0)

    # Result details
    columns = models.SmallIntegerField()
    rows = models.SmallIntegerField()

    # Representations
    def __str__(self):
        if self.shuf == "none":
            return "{}-{} -x- {} to {} via {}, {}, {}mm".format(
                self.parby, self.splby, self.comp, self.tgt, self.algo, self.batch, self.mask
            )
        else:
            return "{}: {}-{} -x- {} to {} via {}, {}, {}mm, seed={}".format(
                self.shuf, self.parby, self.splby, self.comp, self.tgt, self.algo, self.batch, self.mask, self.seed,
            )


class ConnectivityMatrix(models.Model):
    """ Connectivity matrices can be available in the conn directory. """

    path = models.FilePathField(path="/data/conn", max_length=256)
    columns = models.SmallIntegerField()
    rows = models.SmallIntegerField()

    name = models.CharField(max_length=64)
    description = models.TextField()


class ResultSummary(models.Model):
    """ After summarizing all of the result files, cache the results here to save processing. """

    summary_date = models.DateTimeField('date_summarized')
    num_results = models.IntegerField(default=0)
    num_actuals = models.IntegerField(default=0)
    num_shuffles = models.IntegerField(default=0)
    num_splits = models.IntegerField(default=0)

    # @classmethod decorator allows this method to be called on the class, without an instance
    @classmethod
    def empty(cls, timestamp=False):
        if timestamp:
            summary_date = django_timezone.now()
        else:
            summary_date = datetime.datetime.strptime("1900-01-01 00:00:00-0400", "%Y-%m-%d %H:%M:%S%z")
        return cls(
            summary_date=summary_date,
            num_results=0,
            num_actuals=0,
            num_shuffles=0,
            num_splits=0,
        )

    @classmethod
    def current(cls):
        n_all = PushResult.objects.count()
        n_real = PushResult.objects.filter(shuf='none').count()
        return cls(
            summary_date=django_timezone.now(),
            num_results=n_all,
            num_actuals=n_real,
            num_shuffles=n_all - n_real,
            num_splits=0,
        )

    def to_json(self):
        json = "{\n"
        json += "    \"{}\":\"{}\",\n".format("summary_date", self.summary_date.strftime("%m/%d/%Y %H:%M"))
        json += "    \"{}\":\"{:,}\",\n".format("num_results", self.num_results)
        json += "    \"{}\":\"{:,}\",\n".format("num_actuals", self.num_actuals)
        json += "    \"{}\":\"{:,}\",\n".format("num_shuffles", self.num_shuffles)
        json += "    \"{}\":\"{:,}\"\n".format("num_splits", self.num_splits)
        json += "}"
        return json

    class Meta:
        ordering = ["summary_date"]


class GroupedResultSummary(models.Model):
    """ Middle ground between too-broad ResultSummary and too-narrow PushResult """

    summary_date = models.DateTimeField('date_summarized')
    sourcedata = models.CharField(db_index=True, max_length=128, default="")
    descriptor = models.CharField(db_index=True, max_length=16, default="")
    comp = models.CharField(max_length=48)
    parby = models.CharField(max_length=16)
    resample = models.CharField(max_length=16)  # 'whole', 'split-half', 'split-quarter'
    mask = models.CharField(max_length=32)
    num_reals = models.IntegerField(default=0)
    num_agnos = models.IntegerField(default=0)
    num_dists = models.IntegerField(default=0)
    num_be04s = models.IntegerField(default=0)

    summary = models.ForeignKey(ResultSummary, on_delete=models.CASCADE)

    class Meta:
        ordering = ["summary_date"]


class StatusCounts(models.Model):
    """ For a given group, count tsv, json, log, ejgo, etc files. """

    summary_date = models.DateTimeField('date_summarized')
    num_tsvs = models.IntegerField(default=0)
    num_jsons = models.IntegerField(default=0)
    num_logs = models.IntegerField(default=0)
    num_shuffle_maps = models.IntegerField(default=0)
    num_summaries = models.IntegerField(default=0)
    num_entrez_ranks = models.IntegerField(default=0)
    num_ejgo_roc_0002_2048 = models.IntegerField(default=0)
    num_ejgo_roc_0005_0128 = models.IntegerField(default=0)

    summary = models.ForeignKey(GroupedResultSummary, on_delete=models.CASCADE)

    # Representations
    def __str__(self):
        return ":".join([str(x) for x in [
            self.num_tsvs, self.num_jsons, self.num_logs, self.num_shuffle_maps, self.num_summaries,
            self.num_entrez_ranks, self.num_ejgo_roc_0002_2048, self.num_ejgo_roc_0005_0128,
        ]])
