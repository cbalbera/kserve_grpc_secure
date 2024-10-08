# Copyright 2022 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import concurrent.futures
import signal
import sys
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from ray import serve as rayserve
from ray.serve.api import Deployment
from ray.serve.handle import DeploymentHandle

from . import logging
from .constants.constants import (
    DEFAULT_HTTP_PORT,
    DEFAULT_GRPC_PORT,
    MAX_GRPC_MESSAGE_LENGTH,
)
from .logging import logger
from .model import BaseKServeModel
from .model_repository import ModelRepository
from .protocol.dataplane import DataPlane
from .protocol.grpc.server import GRPCServer
from .protocol.model_repository_extension import ModelRepositoryExtension
from .protocol.rest.server import UvicornServer
from .utils import utils
from .api import creds_utils
from kserve.errors import NoModelReady

parser = argparse.ArgumentParser(
    add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# Model Server Arguments: The arguments are passed to the kserve.ModelServer object
parser.add_argument(
    "--http_port",
    default=DEFAULT_HTTP_PORT,
    type=int,
    help="The HTTP Port listened to by the model server.",
)
parser.add_argument(
    "--grpc_port",
    default=DEFAULT_GRPC_PORT,
    type=int,
    help="The GRPC Port listened to by the model server.",
)
parser.add_argument(
    "--workers",
    default=1,
    type=int,
    help="The number of uvicorn workers for multi-processing.",
)
parser.add_argument(
    "--max_threads",
    default=4,
    type=int,
    help="The max number of gRPC processing threads.",
)
parser.add_argument(
    "--max_asyncio_workers",
    default=None,
    type=int,
    help="The max number of asyncio workers to spawn.",
)
parser.add_argument(
    "--enable_grpc",
    default=True,
    type=lambda x: utils.strtobool(x),
    help="Enable gRPC for the model server.",
)
parser.add_argument(
    "--enable_docs_url",
    default=False,
    type=lambda x: utils.strtobool(x),
    help="Enable docs url '/docs' to display Swagger UI.",
)
parser.add_argument(
    "--enable_latency_logging",
    default=True,
    type=lambda x: utils.strtobool(x),
    help="Enable a log line per request with preprocess/predict/postprocess latency metrics.",
)
parser.add_argument(
    "--configure_logging",
    default=True,
    type=lambda x: utils.strtobool(x),
    help="Enable to configure KServe and Uvicorn logging.",
)
parser.add_argument(
    "--log_config_file",
    default=None,
    type=str,
    help="File path containing UvicornServer's log config. Needs to be a yaml or json file.",
)
parser.add_argument(
    "--access_log_format",
    default=None,
    type=str,
    help="The asgi access logging format. It allows to override only the `uvicorn.access`'s format configuration "
    "with a richer set of fields",
)
parser.add_argument(
    "--secure_grpc_server",
    default=False,
    type=bool,
    choices=[True, False],
    help="Enable gRPC server authentication using SSL for the model server.")
parser.add_argument(
    "--ssl_server_key",
    default=None,
    type=str,
    help="File path for SSL server key used for gRPC server authentication.")
parser.add_argument(
    "--ssl_server_cert",
    default=None,
    type=str,
    help="File path for SSL server key used for gRPC server authentication.")
parser.add_argument(
    "--ssl_ca_cert",
    default=None,
    type=str,
    help="File path for SSL server key used for gRPC server authentication.")

# Model arguments: The arguments are passed to the kserve.Model object
parser.add_argument(
    "--model_name",
    default="model",
    type=str,
    help="The name of the model used on the endpoint path.",
)
parser.add_argument(
    "--predictor_host",
    default=None,
    type=str,
    help="The host name used for calling to the predictor from transformer.",
)
# For backwards compatibility.
parser.add_argument(
    "--protocol",
    default="v1",
    type=str,
    choices=["v1", "v2", "grpc-v2"],
    help="The inference protocol used for calling to the predictor from transformer. "
    "Deprecated and replaced by --predictor_protocol",
)
parser.add_argument(
    "--predictor_protocol",
    default="v1",
    type=str,
    choices=["v1", "v2", "grpc-v2"],
    help="The inference protocol used for calling to the predictor from transformer.",
)
parser.add_argument(
    "--predictor_use_ssl",
    default=False,
    type=lambda x: utils.strtobool(x),
    help="Use ssl for the http connection to the predictor.",
)
parser.add_argument(
    "--predictor_request_timeout_seconds",
    default=600,
    type=int,
    help="The timeout seconds for the request sent to the predictor.",
)
parser.add_argument(
    "--grpc_max_send_message_length",
    default=MAX_GRPC_MESSAGE_LENGTH,
    type=int,
    help="The max message length for gRPC send message.",
)
parser.add_argument(
    "--grpc_max_receive_message_length",
    default=MAX_GRPC_MESSAGE_LENGTH,
    type=int,
    help="The max message length for gRPC receive message.",
)
args, _ = parser.parse_known_args()

app = FastAPI(
    title="KServe ModelServer",
    version=metadata.version("kserve"),
    docs_url="/docs" if args.enable_docs_url else None,
    redoc_url=None,
    default_response_class=ORJSONResponse,
)


class ModelServer:
    def __init__(
        self,
        http_port: int = args.http_port,
        grpc_port: int = args.grpc_port,
        workers: int = args.workers,
        max_threads: int = args.max_threads,
        max_asyncio_workers: int = args.max_asyncio_workers,
        registered_models: Optional[ModelRepository] = None,
        enable_grpc: bool = args.enable_grpc,
        enable_docs_url: bool = args.enable_docs_url,
        enable_latency_logging: bool = args.enable_latency_logging,
        access_log_format: str = args.access_log_format,
        secure_grpc_server: bool = args.secure_grpc_server,
        ssl_server_key: Union[str, bytes] = args.ssl_server_key,
        ssl_server_cert: Union[str, bytes] = args.ssl_server_cert,
        ssl_ca_cert: Union[str, bytes] = args.ssl_ca_cert
    ):
        """KServe ModelServer Constructor

        Args:
            http_port: HTTP port. Default: ``8080``.
            grpc_port: GRPC port. Default: ``8081``.
            workers: Number of uvicorn workers. Default: ``1``.
            max_threads: Max number of gRPC processing threads. Default: ``4``
            max_asyncio_workers: Max number of AsyncIO threads. Default: ``None``
            registered_models: A optional Model repository with registered models.
            enable_grpc: Whether to turn on grpc server. Default: ``True``
            enable_docs_url: Whether to turn on ``/docs`` Swagger UI. Default: ``False``.
            enable_latency_logging: Whether to log latency metric. Default: ``True``.
            access_log_format: Format to set for the access log (provided by asgi-logger). Default: ``None``.
                               it allows to override only the `uvicorn.access`'s format configuration with a richer
                               set of fields (output hardcoded to `stdout`). This limitation is currently due to the
                               ASGI specs that don't describe how access logging should be implemented in detail
                               (please refer to this Uvicorn
                               [github issue](https://github.com/encode/uvicorn/issues/527) for more info).
            secure_grpc_server: Whether to enable secure grpc server. Default: ``False``.
            ssl_server_key: File path or contents to server key for secure grpc server credentials. Default: ``None``.
            ssl_server_cert: File path or contents to server cert for secure grpc server credentials. Default: ``None``.
            ssl_ca_cert: File path or contents to CA cert for secure grpc server credentials. Default: ``None``.
        """
        self.registered_models = (
            ModelRepository() if registered_models is None else registered_models
        )
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.workers = workers
        self.max_threads = max_threads
        self.max_asyncio_workers = max_asyncio_workers
        self.enable_grpc = enable_grpc
        self.enable_docs_url = enable_docs_url
        self.enable_latency_logging = enable_latency_logging
        self.dataplane = DataPlane(model_registry=self.registered_models)
        self.model_repository_extension = ModelRepositoryExtension(
            model_registry=self.registered_models
        )
        self._grpc_server = None
        self._rest_server = None
        self.secure_grpc_server = secure_grpc_server
        self.grpc_ssl_key = ssl_server_key
        self.grpc_ssl_cert = ssl_server_cert
        self.grpc_ssl_ca_cert = ssl_ca_cert
        if self.enable_grpc:
            if self.secure_grpc_server:
                server_credentials = []
                ssl_key = creds_utils.parse_grpc_server_credentials(self.grpc_ssl_key)
                server_credentials.append(ssl_key)
                ssl_cert = creds_utils.parse_grpc_server_credentials(self.grpc_ssl_cert)
                server_credentials.append(ssl_cert)
                ssl_ca_cert = creds_utils.parse_grpc_server_credentials(self.grpc_ssl_ca_cert)
                server_credentials.append(ssl_ca_cert)
                self._grpc_server = GRPCServer(
                    grpc_port,
                    self.dataplane,
                    self.model_repository_extension,
                    kwargs=vars(args),
                    secure_server=self.secure_grpc_server,
                    grpc_secure_server_credentials=server_credentials,
                )
            else:
                self._grpc_server = GRPCServer(
                    grpc_port,
                    self.dataplane,
                    self.model_repository_extension,
                    kwargs=vars(args),
                )
        if args.configure_logging:
            # If the logger does not have any handlers, then the logger is not configured.
            # For backward compatibility, we configure the logger here.
            if len(logger.handlers) == 0:
                logging.configure_logging(args.log_config_file)
        self.access_log_format = access_log_format
        self._custom_exception_handler = None

    async def _serve_rest(self):
        logger.info(f"Starting uvicorn with {self.workers} workers")
        loop = asyncio.get_event_loop()
        if sys.platform not in ["win32", "win64"]:
            sig_list = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]
        else:
            sig_list = [signal.SIGINT, signal.SIGTERM]

        for sig in sig_list:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.stop(sig=s))
            )
        if self._custom_exception_handler is None:
            loop.set_exception_handler(self.default_exception_handler)
        else:
            loop.set_exception_handler(self._custom_exception_handler)
        self._rest_server = UvicornServer(
            app,
            self.http_port,
            self.dataplane,
            self.model_repository_extension,
            # By setting log_config to None we tell Uvicorn not to configure logging as it is already
            # configured by kserve.
            log_config=None,
            access_log_format=self.access_log_format,
            workers=self.workers,
        )
        await self._rest_server.run()

    def start(
        self, models: Union[List[BaseKServeModel], Dict[str, Deployment]]
    ) -> None:
        """Start the model server with a set of registered models.

        Args:
            models: a list of models to register to the model server.
        """
        if isinstance(models, list):
            at_least_one_model_ready = False
            for model in models:
                if isinstance(model, BaseKServeModel):
                    if model.ready:
                        at_least_one_model_ready = True
                        self.register_model(model)
                        # pass whether to log request latency into the model
                        model.enable_latency_logging = self.enable_latency_logging
                else:
                    raise RuntimeError("Model type should be 'BaseKServeModel'")
            if not at_least_one_model_ready and models:
                raise NoModelReady(models)
        elif isinstance(models, dict):
            if all([isinstance(v, Deployment) for v in models.values()]):
                # TODO: make this port number a variable
                rayserve.start(
                    detached=True, http_options={"host": "0.0.0.0", "port": 9071}
                )
                for key in models:
                    models[key].deploy()
                    handle = models[key].get_handle()
                    self.register_model_handle(key, handle)
            else:
                raise RuntimeError("Model type should be RayServe Deployment")
        else:
            raise RuntimeError("Unknown model collection types")

        if self.max_asyncio_workers is None:
            # formula as suggest in https://bugs.python.org/issue35279
            self.max_asyncio_workers = min(32, utils.cpu_count() + 4)
        logger.info(f"Setting max asyncio worker threads as {self.max_asyncio_workers}")
        asyncio.get_event_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self.max_asyncio_workers)
        )

        async def servers_task():
            servers = [self._serve_rest()]
            if self.enable_grpc:
                servers.append(self._grpc_server.start(self.max_threads))
            await asyncio.gather(*servers)

        asyncio.run(servers_task())

    async def stop(self, sig: Optional[int] = None):
        """Stop the instances of REST and gRPC model servers.

        Args:
            sig: The signal to stop the server. Default: ``None``.
        """
        logger.info("Stopping the model server")
        if self._rest_server:
            logger.info("Stopping the rest server")
            await self._rest_server.stop()
        if self._grpc_server:
            logger.info("Stopping the grpc server")
            await self._grpc_server.stop(sig)
        for model_name in list(self.registered_models.get_models().keys()):
            self.registered_models.unload(model_name)

    def register_exception_handler(
        self,
        handler: Callable[[asyncio.events.AbstractEventLoop, Dict[str, Any]], None],
    ):
        """Add a custom handler as the event loop exception handler.

        If a handler is not provided, the default exception handler will be set.

        handler should be a callable object, it should have a signature matching '(loop, context)', where 'loop'
        will be a reference to the active event loop, 'context' will be a dict object (see `call_exception_handler()`
        documentation for details about context).
        """
        self._custom_exception_handler = handler

    def default_exception_handler(
        self, loop: asyncio.events.AbstractEventLoop, context: Dict[str, Any]
    ):
        """Default exception handler for event loop.

        This is called when an exception occurs and no exception handler is set.
        This can be called by a custom exception handler that wants to defer to the default handler behavior.
        """
        if "exception" in context:
            logger.error(f"Caught exception: {context.get('exception')}")
        logger.error(f"message: { context.get('message')}")
        loop.default_exception_handler(context)

    def register_model_handle(self, name: str, model_handle: DeploymentHandle):
        """Register a model handle to the model server.

        Args:
            name: The name of the model handle.
            model_handle: The model handle object.
        """
        self.registered_models.update_handle(name, model_handle)
        logger.info("Registering model handle: %s", name)

    def register_model(self, model: BaseKServeModel):
        """Register a model to the model server.

        Args:
            model: The model object.
        """
        if not model.name:
            raise Exception("Failed to register model, model.name must be provided.")
        self.registered_models.update(model)
        logger.info("Registering model: %s", model.name)
