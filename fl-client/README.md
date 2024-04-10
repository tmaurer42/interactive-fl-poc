# FL API

This is the client application for interactive federated machine learning.

## Requirements

- Docker
- Node Version >= 20
- Yarn
- Python > 3.12 for development


## Start / Debug Client

Create a virtual Python environment with version > 3.12 and install the requirements.

Run ```yarn install```

Use the Debug configuration "Client: Debug (FL Client)" in the VSCode Debug section. Debugging the Client this way supports hot reload during development.

When changing the TypeScript files in the src folder while debugging, run ```yarn build```. This will immediately reflect the changes in the running application without restarting.

## Test

### Flask App
Use "Client: Test Debug" in the VSCode Debug section.

### TypesScript modules
TypesScript modules can be debugged directly in the browser.
