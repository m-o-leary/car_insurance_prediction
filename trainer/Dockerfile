FROM python:3
RUN apt-get -y install git  
RUN mkdir app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt \
    && pip install -e .
