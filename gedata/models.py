from __future__ import absolute_import, unicode_literals

from django.db import models
import datetime

# Create your models here.
class PushResult(models.Model):
    """ Each run of PyGEST has many features. They are recorded in the PushResult class. """

    # File information
    json_path = models.FilePathField(path="/data", max_length=256)
    tsv_path = models.FilePathField(path="/data", max_length=256)
    log_path = models.FilePathField(path="/data", max_length=256)

    # Execution details
    start_date = models.DateTimeField('date_started')
    end_date = models.DateTimeField('date_completed')
    host = models.CharField(max_length = 16)
    command = models.CharField(max_length = 256)
    version = models.CharField(max_length = 16)
    # duration is in seconds
    duration = models.IntegerField()

    # Data preparation and execution details
    shuffle = models.CharField(max_length = 16)  # 'derivatives': 'derivatives', 'agnostic': 'shuffles', 'distance': 'distshuffles', 'edges': 'edgeshuffles'
    subject = models.CharField(max_length = 32)
    hemisphere = models.CharField(max_length = 1)
    cortex = models.CharField(max_length = 3)
    probes = models.CharField(max_length = 16)
    target = models.CharField(max_length = 3)
    algorithm = models.CharField(max_length = 8)
    normalization = models.CharField(max_length = 16)
    comparator = models.CharField(max_length = 32)
    mask = models.CharField(max_length = 32)
    adjustment = models.CharField(max_length = 16)
    seed = models.IntegerField()

    # Result details
    columns = models.SmallIntegerField()
    rows = models.SmallIntegerField()

    # Representations
    def __str__(self):
        if self.shuffle == "derivatives":
            return "{} -x- {} to {} via {}".format(
                self.subject, self.comparator, self.target, self.algorithm
            )
        else:
            return "{}: {} -x- {} to {} via {}, seed={}".format(
                self.shuffle, self.subject, self.comparator, self.target, self.algorithm, self.seed,
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
    num_results = models.IntegerField()
    num_actuals = models.IntegerField()
    num_shuffles = models.IntegerField()
    num_distshuffles = models.IntegerField()
    num_edgeshuffles = models.IntegerField()
    num_splits = models.IntegerField()

    @classmethod
    def empty(cls):
        return cls(
            summary_date = datetime.datetime.strptime("1900-01-01 00:00:00-0400", "%Y-%m-%d %H:%M:%S%z"),
            num_results = 0,
            num_actuals = 0,
            num_shuffles = 0,
            num_distshuffles = 0,
            num_edgeshuffles = 0,
            num_splits = 0,
        )
    class Meta:
        ordering = ["summary_date"]