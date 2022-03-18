FROM python:3.8
COPY main.py /
COPY modelCIFAR.h5 /
COPY requirements.txt /
RUN pip install -r requirements.txt
#Следующие 2 строки для работы cv2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD python ./main.py
