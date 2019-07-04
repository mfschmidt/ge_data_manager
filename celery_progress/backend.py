from abc import ABCMeta, abstractmethod
from decimal import Decimal

from celery.result import AsyncResult

PROGRESS_STATE = 'PROGRESS'


class AbstractProgressRecorder(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_progress(self, current, total):
        pass


class ConsoleProgressRecorder(AbstractProgressRecorder):

    def set_progress(self, current, total):
        print('processed {} items of {}'.format(current, total))


class ProgressRecorder(AbstractProgressRecorder):

    def __init__(self, task):
        self.task = task

    def set_progress(self, current, total):
        percent = 0
        if total > 0:
            percent = (Decimal(current) / Decimal(total)) * Decimal(100)
            percent = float(round(percent, 2))
        self.task.update_state(
            state=PROGRESS_STATE,
            meta={
                'current': current,
                'total': total,
                'percent': percent,
            }
        )


class Progress(object):

    def __init__(self, task_id):
        self.task_id = task_id
        self.result = AsyncResult(task_id)
        self.current = 0
        self.total = 100

    def get_info(self):
        if self.result.ready():
            return {
                'complete': True,
                'success': self.result.successful(),
                'progress': {'current': self.total, 'total': self.total, 'percent': 100,},
            }
        elif self.result.state == PROGRESS_STATE:
            info = self.result.info
            self.current = info['current']
            self.total = info['total']
            return {
                'complete': False,
                'success': None,
                'progress': info,
            }
        elif self.result.state in ['PENDING', 'STARTED']:
            return {
                'complete': False,
                'success': None,
                'progress': {'current': self.current, 'total': self.total, 'percent': 0,},
            }
        return self.result.info


