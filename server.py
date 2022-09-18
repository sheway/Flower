import flwr as fl
import os

# os.environ['GRPC_TRACE'] = 'all'
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

# Start Flower server
fl.server.start_server(
    server_address="localhost:7001",
    config={"num_rounds": 3},
)