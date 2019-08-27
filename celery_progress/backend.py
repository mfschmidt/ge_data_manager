from abc import ABCMeta, abstractmethod
from decimal import Decimal

from celery.result import AsyncResult

PROGRESS_STATE = 'PROGRESS'


class AbstractProgressRecorder(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_progress(self, current, total, message=""):
        pass


class ConsoleProgressRecorder(AbstractProgressRecorder):

    def set_progress(self, current, total, message=""):
        print('processed {} items of {} ({})'.format(current, total, message))



class ProgressRecorder(AbstractProgressRecorder):

    def __init__(self, task, message=""):
        self.task = task
        self.message = message

    def set_progress(self, current, total, message=""):
        percent = 0
        if total > 0:
            percent = (Decimal(current) / Decimal(total)) * Decimal(100)
            percent = float(round(percent, 2))
        if message != "":
            self.message = message
        self.task.update_state(
            state=PROGRESS_STATE,
            meta={
                'current': current,
                'total': total,
                'percent': percent,
                'message': self.message,
            }
        )


class Progress(object):

    def __init__(self, task_id, message=""):
        self.task_id = task_id
        self.result = AsyncResult(task_id)
        self.current = 0
        self.total = 100
        self.message = message

    def get_info(self):
        if self.result.ready():
            return {
                'complete': True,
                'success': self.result.successful(),
                'progress': {'current': self.total, 'total': self.total, 'percent': 100,},
                'message': "complete",
            }
        elif self.result.state == PROGRESS_STATE:
            info = self.result.info
            self.current = info['current']
            self.total = info['total']
            if 'message' in info:
                self.message = info['message']
            return {
                'complete': False,
                'success': None,
                'progress': info,
                'message': self.message,
            }
        elif self.result.state in ['PENDING', 'STARTED']:
            return {
                'complete': False,
                'success': None,
                'progress': {'current': self.current, 'total': self.total, 'percent': 0,},
                'message': self.message,
            }
        return self.result.info


