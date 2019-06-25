from django.db import models
import datetime

# Create your models here.
class PushResult(models.Model):
    """ Each run of PyGEST has many features. They are recorded in the PushResult class. """

    # File information
    json_path = models.FilePathField()
    tsv_path = models.FilePathField()
    log_path = models.FilePathField()

    # Execution details
    start_date = models.DateTimeField('date_started')
    end_date = models.DateTimeField('date_completed')
    host = models.CharField(max_length = 16)
    command = models.CharField(max_length = 16)
    version = models.CharField(max_length = 16)
    duration = models.IntegerField()

    # Data preparation and execution details
    shuffle = models.CharField(max_length = 16)  # 'none': 'derivatives', 'agnostic': 'shuffles', 'distance': 'distshuffles', 'edges': 'edgeshuffles'
    subject = models.CharField(max_length = 32)
    hemisphere = models.CharField(max_length = 1)
    cortex = models.CharField(max_length = 3)
    probes = models.CharField(max_length = 16)
    target = models.CharField(max_length = 3)
    algorithm = models.CharField(max_length = 8)
    normalization = models.CharField(max_length = 16)
    comparator = models.CharField(max_length = 32)
    mask = models.CharField(max_length = 8)
    adjustment = models.CharField(max_length = 16)
    seed = models.CharField(max_length = 8)

    # Result details
    columns = models.SmallIntegerField()
    rows = models.SmallIntegerField()

    # Representations
    def __str__(self):
        if self.shuffle == "none":
            return "{}-{}-{} ({})".format(
                self.subject, self.comparator, self.target, self.algorithm
            )
        else:
            return "{}-{}-{} ({}, {:3}-{})".format(
                self.subject, self.comparator, self.target, self.algorithm, self.seed, self.shuffle
            )


class ConnectivityMatrix(models.Model):
    """ Connectivity matrices can be available in the conn directory. """

    path = models.FilePathField()
    columns = models.SmallIntegerField()
    rows = models.SmallIntegerField()

    name = models.CharField(max_length = 64)
    description = models.TextField()
