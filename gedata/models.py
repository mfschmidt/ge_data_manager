from __future__ import absolute_import, unicode_literals

from django.db import models
from django.utils import timezone as django_timezone
import datetime


# Create your models here.
class PushResult(models.Model):
    """ Each run of PyGEST has many features. They are recorded in the PushResult class. """

    # Index for rapid searching of inventory
    sourcedata = models.CharField(db_index=True, max_length = 128, default="")
    descriptor = models.CharField(db_index=True, max_length = 16, default="")

    # File information
    json_path = models.FilePathField(path="/data", max_length=256)
    tsv_path = models.FilePathField(path="/data", max_length=256)
    log_path = models.FilePathField(path="/data", max_length=256)

    # Execution details
    start_date = models.DateTimeField('date_started')
    end_date = models.DateTimeField('date_completed')
    host = models.CharField(max_length = 16)
    command = models.CharField(max_length = 384)
    version = models.CharField(max_length = 64)  # 'running uninstalled without version' longest
    # duration is in seconds
    duration = models.IntegerField()

    # Data preparation and execution details
    shuf = models.CharField(db_index=True, max_length = 16)  # 'none', 'agno', 'be04', 'dist', etc.
    resample = models.CharField(max_length = 16)  # 'whole', 'split-half', 'split-quarter'
    sub = models.CharField(max_length = 32)
    hem = models.CharField(max_length = 1)
    samp = models.CharField(max_length = 16)
    prob = models.CharField(max_length = 16)
    parby = models.CharField(max_length = 16)
    splby = models.CharField(max_length = 16)
    batch = models.CharField(max_length = 16)
    tgt = models.CharField(max_length = 3)
    algo = models.CharField(max_length = 8)
    norm = models.CharField(max_length = 16)
    comp = models.CharField(max_length = 48)
    mask = models.CharField(max_length = 32)
    adj = models.CharField(max_length = 16)
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

    name = models.CharField(max_length = 64)
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
            summary_date = summary_date,
            num_results = 0,
            num_actuals = 0,
            num_shuffles = 0,
            num_splits = 0,
        )

    @classmethod
    def current(cls):
        n_all = PushResult.objects.count()
        n_real = PushResult.objects.filter(shuf='none').count()
        return cls(
            summary_date = django_timezone.now(),
            num_results = n_all,
            num_actuals = n_real,
            num_shuffles = n_all - n_real,
            num_splits = 0,
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
