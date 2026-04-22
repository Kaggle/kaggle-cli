# This Docker image is a modified version of the python image that
# includes [hatch](https://hatch.pypa.io/latest/)

ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

RUN python -m pip install --upgrade pip

RUN python -m pip install hatch==1.16.5

ENTRYPOINT ["hatch"]