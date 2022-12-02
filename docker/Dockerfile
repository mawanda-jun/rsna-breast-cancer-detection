# Starting image
FROM nvcr.io/nvidia/pytorch:22.08-py3

#################################################################

# Useful instructions if base image is nvcr.io
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y \
        && apt install software-properties-common -y \
        && apt-get update -y 
        
# Other dependencies
RUN apt update -y \
    && apt install -y \
        tmux \
        htop
        
# Install ffmpeg for Fiftyone
RUN apt update -y \ 
    && apt install -y ffmpeg 

# From now on run container as a non-root user
ARG UNAME=mawanda
ARG GNAME=mawanda
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $GNAME
RUN useradd -m -u $UID -g $GID -G $GID -o -s /bin/bash $UNAME
USER $UNAME

# RUN useradd -u 1002 -g 1004 --create-home --shell /bin/bash baseuser
# USER baseuser
WORKDIR /home/$UNAME
# Add user scripts to global environment
ENV PATH=/home/$UNAME/.local/bin:${PATH}

# Install Tensorflow & models
# RUN pip install --user tensorflow tf-models-official

# Install every package found in requirements.txt
COPY --chown=$UNAME:$GNAME requirements.txt requirements.txt
RUN pip install --user -r requirements.txt && rm requirements.txt

# Fix git name and email
RUN git config --global user.name mawanda-jun && git config --global user.email giovanni.cavallin@outlook.com

ENV PYTHONPATH .
