FROM python:3
RUN apt-get -y install git  
RUN mkdir app
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
