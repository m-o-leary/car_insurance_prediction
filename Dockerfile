FROM python:3
RUN apt-get -y install git  
RUN mkdir app
ADD . /app
WORKDIR /app
# TODO: This should replace the last line once dev is complete on trainer_lib
RUN pip install /app/ && pip install -r requirements.txt
    
# RUN pip install -r requirements.txt
