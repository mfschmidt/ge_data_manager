FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install --upgrade git+git://github.com/mfschmidt/PyGEST.git

COPY manage.py /code/
COPY celery_progress /code/celery_progress
COPY gedata /code/gedata
COPY ge_data_manager /code/ge_data_manager

