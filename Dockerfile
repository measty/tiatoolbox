# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /tiatoolbox
RUN apt-get update && apt-get -y install gcc
RUN apt-get install -y python3-opencv
RUN apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install bokeh
RUN pip3 install flask-cors

COPY . .
RUN python setup.py install
EXPOSE 5000 5006

FROM nginx
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

CMD [ "bokeh", "serve", "./tiatoolbox/visualization/render_demo" ]

