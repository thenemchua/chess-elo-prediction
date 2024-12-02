# create an image from an environment
FROM python:3.10.6-buster

# COPY -> select the folder you need
COPY api api
COPY requirements.txt requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# install your package
COPY model model
COPY setup.py setup.py
RUN pip install .

EXPOSE 8000

# controls functinnality of the container
# uvicorn to control the server port
# local

CMD uvicorn api.app_file:app --host 0.0.0.0

# Deploy to GCP
#CMD uvicorn api.app_file:app --host 0.0.0.0 --port 8000
