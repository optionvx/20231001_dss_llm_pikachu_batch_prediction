# Dockerfile: Please note it's not advised to run a Docker image as root when running pipeline code.
# Please create a new user to better fit security needs after using root priviledges for necassary installations.


FROM python:3.10.12
USER root

WORKDIR /root

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

COPY your_gcp_creds.json /root

ENV GOOGLE_APPLICATION_CREDENTIALS=/root/your_gcp_creds.json

RUN echo '[Credentials]\ngs_service_key_file = /root/your_gcp_creds.json' \
    > /etc/boto.cfg

COPY your_gcp_creds.json $GOOGLE_APPLICATION_CREDENTIALS


COPY pipeline.py /root
COPY llm_label_flow.py /root
COPY pipeline_component.py /root
COPY utils.py /root
COPY requirements.txt /root

RUN pip install --upgrade pip==23.2.1
RUN pip install -r /root/requirements.txt
RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

RUN chmod +x /root/pipeline.py

ENTRYPOINT ["python","pipeline.py"]
