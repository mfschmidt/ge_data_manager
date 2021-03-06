""" This implementation of celery within django 2 is courtesy of the official celery documentation
    at https://docs.celeryproject.org/en/latest/django/first-steps-with-django.html. """

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set the default django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ge_data_manager.settings")
app = Celery('ge_data_manager')

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print("Request: {0!r}".format(self.request))

def celery_status():
    return len(celery_workers())

def celery_tasks():
    return list(app.control.inspect().active().keys())

def celery_workers():
    return list(app.control.inspect().stats().keys())

def celery_running(job_name):
    try:
        task_dict = app.control.inspect().active()
    except ConnectionResetError:
        task_dict = app.control.inspect().active()

    if task_dict is None:
        return None

    if len(task_dict) > 0:
        for k in task_dict.keys():
            if job_name in task_dict[k]:
                return True

    return False

def celery_id_from_name(job_name):
    try:
        task_dict = app.control.inspect().active()
    except ConnectionResetError:
        task_dict = app.control.inspect().active()

    if task_dict is None:
        return None

    if len(task_dict) > 0:
        for k in task_dict.keys():
            for d in task_dict[k]:
                if job_name in d['name']:
                    return d['id']

    """ One example of a task_dict:
    [{
        'id': '97b8c300-9193-4834-9a04-0472997e5d54',
        'name': 'gedata.tasks.build_plot',
        'args': "('hcpgw64s',)",
        'kwargs': "{'data_path': '/data'}",
        'type': 'gedata.tasks.build_plot',
        'hostname': 'celery@mq',
        'time_start': 1566873261.7720587,
        'acknowledged': True,
        'delivery_info': {
            'exchange': '',
            'routing_key': 'celery',
            'priority': 0,
            'redelivered': False
        },
        'worker_pid': 26
    }]
    """
    return None


def celery_tasks_in_progress():
    try:
        task_dict = app.control.inspect().active()
    except ConnectionResetError:
        task_dict = app.control.inspect().active()

    tasks = []

    if task_dict is None:
        return tasks

    if len(task_dict) > 0:
        for k in task_dict.keys():
            for d in task_dict[k]:
                print("  running task named {}".format(d['name']))
                if 'build_plot' in d['name']:
                    tasks.append("train_test_{}.png".format(d['args'][2:10].lower()))

    return tasks
