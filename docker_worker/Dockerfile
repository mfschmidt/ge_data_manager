FROM python:3

ENV PYTHONUNBUFFERED 1
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

RUN apt-get update && apt-get install -y openjdk-11-jre && apt-get install -y ant && apt-get clean
RUN export JAVA_HOME

RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install --upgrade git+git://github.com/mfschmidt/PyGEST.git

COPY manage.py /code/
COPY celery_progress /code/celery_progress
COPY gedata /code/gedata
COPY ge_data_manager /code/ge_data_manager

# RUN groupadd -g 1000 workeruser && useradd -u 1000 -g workeruser workeruser && chmod 777 /code/gedata/static/gedata
# USER workeruser
