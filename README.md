# FL POC

This is a proof of concept application for interactive federated machine learning.

## Recommended Code Editor

This project is optimized for development in Visual Studio Code. Open the the workspace file **fl_poc.code-workspace** for the best development experience.

## Prerequisites

- Docker
- Python > 3.12 for development

## Run Application

```
docker compose -f docker-compose.yml up --build
```

### Debug

- Start the application via docker compose:

```
docker compose -f docker-compose.debug.yml up --build
```

- Attach the debugger for the server by selecting "server: Attach" in the VSCode Debug section. The server will start as soon as the debugger is attached.
- Attach the debugger for the Client App by selecting "Client: Attach" in the VSCode Debug section. The Client App will start as soon as the debugger is attached.

The server will be available at port **5002**.
The Client App will be available at port **4002**.

If you need hot reload, debug the respective project separately.
