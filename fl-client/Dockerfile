# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12.0b4-slim as build

# Install Node.js and typescript dependencies
RUN apt-get update && apt-get install -y curl nodejs npm && \
    curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g typescript webpack webpack-cli ts-loader yarn

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install

COPY . .

# Compile TypeScript with webpack
RUN yarn build


# Production image
FROM python:3.12.0b4-slim

WORKDIR /app

# Copy app folder
COPY --from=build /app /app

# Copy installed python modules
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 6789 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Keeps Python from generating .pyc files in the container / Turns off buffering for easier container logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

EXPOSE 4002

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:4002", "app:app"]
