# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /tiatoolbox
RUN apt-get update && apt-get -y install gcc
RUN apt-get install -y python3-opencv curl
RUN apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install bokeh
RUN pip3 install flask-cors

COPY . .
RUN python setup.py install
EXPOSE 5100

CMD [ "bokeh", "serve", "./tiatoolbox/visualization/render_demo", "--allow-websocket-origin=localhost:5100", "--allow-websocket-origin=iguana.dcs.warwick.ac.uk", "--allow-websocket-origin=tia-web-01.dcs.warwick.ac.uk:5100", "--allow-websocket-origin=20.0.0.9:5100", "--port", "5100", "--use-xheaders", "--unused-session-lifetime", "1000", "--check-unused-sessions", "1000" , "--args", "/app_data"]

