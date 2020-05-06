# imadelh/opencv_tf:full -> contains all requirements and weights for 3 models and ready to use for API
# imadelh/opencv_tf:base -> contains all requirements to run notebooks (weights will be downloaded from the notebook).
# see ./docker-base-requirements for details about requirements and docker image

FROM imadelh/opencv_tf:full
CMD gunicorn --bind 0.0.0.0:8080 wsgi:app