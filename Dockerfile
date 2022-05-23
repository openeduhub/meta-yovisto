FROM python:3.9.13-slim-bullseye
WORKDIR /app
# Install Poetry
RUN apt-get update -y  && apt-get upgrade -y && apt-get update \
    && apt-get install -y ca-certificates curl \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.8/get-poetry.py | POETRY_HOME=/opt/poetry python \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry \
    && poetry config virtualenvs.create false
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install --no-root
RUN python -c "import nltk;nltk.download('stopwords')"
COPY src/app /app
