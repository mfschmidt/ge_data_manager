FROM python:3

ENV PYTHONUNBUFFERED 1
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN apt-get update && apt-get install -y openjdk-8-jre && apt-get install -y ant && apt-get clean
RUN export JAVA_HOME

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --upgrade git+git://github.com/mfschmidt/PyGEST.git
