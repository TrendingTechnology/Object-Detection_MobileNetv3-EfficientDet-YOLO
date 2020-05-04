# Resulting image is available at  imadelh/opencv_tf:full
# It contains all weights and models ready to use.
FROM imadelh/opencv_tf:prod

ADD . /app
WORKDIR /app/app

ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /app

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# CMD gunicorn --threads=4 --bind 0.0.0.0:8080 wsgi:app