FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /api/app
## Install our own lib
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt