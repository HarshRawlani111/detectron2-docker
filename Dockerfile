FROM python:3.8-slim-buster

COPY . /app
WORKDIR /app
#since we are using the COPY command we dont need to ADD files individually
#ADD requirements.txt /
#ADD voc2coco.py /
#ADD jan27_detectron2_docker_v01.py /
#ADD model_final.pth /
#ADD index.html /
#ADD result.html /
#ADD prediction.py /
#ADD flask_app.py /
#ADD inputImage.jpg /
#ADD color_img.jpg /
#ADD config.yml /


ENV PYTHONUNBUFFERED=1

EXPOSE 5000

RUN apt-get update -y 

#RUN pip install -r requirements.txt   
    

#RUN pip install -r requirements.txt

CMD ["python", "flask_app.py"]




