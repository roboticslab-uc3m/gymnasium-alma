FROM ubuntu:22.04

ARG SSL_DEBFILE="libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb"
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Europe/Minsk

COPY . /alma-gymnasium

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get purge -y --autoremove software-properties-common gpg-agent && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        python3.11 \
        libgomp1 \
    && \
    ln -fs /usr/bin/python3.11 /usr/bin/python3 && \
    wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/$SSL_DEBFILE && \
    dpkg -i $SSL_DEBFILE && \
    rm $SSL_DEBFILE && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cffi && \
    pip install --no-cache-dir -e /alma-gymnasium/bandit && \
    pip install --no-cache-dir -e /alma-gymnasium/fakeironing && \
    pip install --no-cache-dir -e /alma-gymnasium/gridworld && \
    mkdir /playground

WORKDIR /playground
