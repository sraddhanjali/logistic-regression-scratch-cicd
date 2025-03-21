# set up python 3.12 to be used as base image
FROM python:3.12

# set the work directory
WORKDIR /app

# copy the project files
COPY . .

RUN rm -rf .venv

# setup environment variables for virtualenv
ENV PYTHONPATH=/app

# activate virtual environment inside the container
RUN python -m venv .venv
RUN /bin/bash .venv/bin/activate

# install dependencies inside the virtual env
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# run the full coverage test
CMD ["make", "full_coverage"]