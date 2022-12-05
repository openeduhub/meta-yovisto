FROM python:3.10
WORKDIR /app

# Configure Poetry
ENV POETRY_VERSION=1.2.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}
# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

RUN poetry config virtualenvs.create false
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install --no-root
RUN python -v
RUN pip3 install nltk
RUN python3 -c "import nltk;nltk.download('stopwords')"
COPY src/app /app
COPY src/data /app/data
