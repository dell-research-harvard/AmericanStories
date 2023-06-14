FROM ubuntu:18.04

WORKDIR /img2txt_pipeline

ARG SSH_PRIVATE_KEY
ARG RCLONE_CONF

# linux setup
RUN apt -y update && apt -y upgrade
RUN apt install -y software-properties-common build-essential curl wget git zip unzip
RUN apt-get install -y ffmpeg libsm6 libxext6


# python setup
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt -y update
RUN apt install -y python3.8 python3.8-dev
RUN apt install -y python3-pip
RUN python3.8 -m pip install --upgrade pip

# assemble codebase and essential files
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN mkdir -p /root/.config/rclone/
RUN echo "${RCLONE_CONF}" > /root/.config/rclone/rclone.conf
RUN curl https://rclone.org/install.sh | bash
RUN rclone copy -P --config /root/.config/rclone/rclone.conf remote:img2txt_pipeline_essentials /img2txt_pipeline/
RUN GIT_SSH_COMMAND='ssh -i /root/.ssh/id_rsa' git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline
RUN cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/
RUN rm -rf /img2txt_pipeline/end-to-end-pipeline

# python dependencies
RUN python3.8 -m pip install -r requirements.txt
