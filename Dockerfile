############################################################
# Dockerfile to build docker image for SUMMA models
############################################################

# Set base image
FROM continuumio/anaconda3:4.2.0

#Maintainer
MAINTAINER Abiola Obamuyide

# Update repository sources list
RUN apt-get update

#install Tensorflow
RUN pip install tensorflow==1.4.1

RUN apt-get install -y python-dev
RUN apt-get install -y build-essential

#install python packages
RUN pip install urllib3
RUN pip install certifi
RUN pip install ordered_set
RUN pip install tqdm

RUN pip install -U nltk
RUN python -c "import nltk ; nltk.download('punkt')"


COPY . /summa
ENV PYTHONPATH /summa
EXPOSE 7009

ENTRYPOINT ["python","/summa/api/startup.py"]
#Done
