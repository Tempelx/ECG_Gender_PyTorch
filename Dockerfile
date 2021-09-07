FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /
WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

COPY . /

ENTRYPOINT [ "python3" ]

CMD ["main.py"]
