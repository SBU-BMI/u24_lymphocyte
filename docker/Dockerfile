FROM nvidia/cuda:8.0-cudnn5-devel 
MAINTAINER Tahsin Kurc

RUN 	apt-get -y update && apt-get -y install wget libfontconfig1 libxrender1 libxtst6 

ENV 	PATH=/root/miniconda2/bin:$PATH
COPY    . /root/u24_lymphocyte/.
RUN 	wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
	sh Miniconda2-latest-Linux-x86_64.sh -b && \
	pip install --upgrade pip && \
	conda install -c conda-forge -c bioconda --yes --file /root/u24_lymphocyte/docker/requirements.txt && \
	conda install --yes -c conda-forge libiconv pygpu==0.6.5 && \
 	pip install cython msgpack openslide-python theano==0.9.0 && \
 	pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip && \
 	cp /root/u24_lymphocyte/docker/run_*.sh /usr/bin && chmod 0755 /usr/bin/run_*.sh && \
	mv /root/u24_lymphocyte/docker/theanorc /root/.theanorc

CMD ["/bin/bash"]
