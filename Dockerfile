FROM python:3.9.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash arturo_docker
USER arturo_docker

WORKDIR /home/arturo_docker
COPY --chown=arturo_docker:arturo_docker requirements.txt requirements.txt
ENV PATH "$PATH:/home/arturo_docker/.local/bin"
RUN pip install --user -r requirements.txt
ENV PATH "$PATH:/home/arturo_docker/.local/bin"
ENV PYTHONPATH "${PYTHONPATH}:/home/arturo_docker/mount/Documents/udacity/MLEng/Module1/clean_churn/tests:/home/arturo_docker/mount/Documents/udacity/MLEng/Module1/clean_churn:"
