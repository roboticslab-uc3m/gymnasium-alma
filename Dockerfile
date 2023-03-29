FROM ubuntu:22.04

ARG SSL_DEBFILE="libssl1.1_1.1.1f-1ubuntu2.17_amd64.deb"
ARG DEBIAN_FRONTEND="noninteractive"

COPY . /alma-playground

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        python3 \
        python3-pip \
        libgomp1 \
    && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/$SSL_DEBFILE && \
    dpkg -i $SSL_DEBFILE && \
    rm $SSL_DEBFILE && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cffi && \
    pip install --no-cache-dir -e /alma-playground/bandit && \
    pip install --no-cache-dir -e /alma-playground/fakeironing && \
    pip install --no-cache-dir -e /alma-playground/gridworld && \
    mkdir /playground

WORKDIR /playground
