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

import base64
import json
import os

import pytest
from kserve import (
    KServeClient,
    V1beta1InferenceService,
    V1beta1InferenceServiceSpec,
    V1beta1PredictorSpec,
    V1beta1TransformerSpec,
    constants,
    InferInput, 
    InferRequest,
    InferenceGRPCClient
)
from kubernetes.client import V1ResourceRequirements
from kubernetes import client
from kubernetes.client import V1Container, V1ContainerPort
from ..common.utils import KSERVE_TEST_NAMESPACE, predict_isvc, predict_grpc
from typing import List
import asyncio
import grpc
from os import path


# gRPC client setup
async def grpc_infer_request(integer: int, port: str, ssl: bool, creds: List, channel_args: any):
    if ssl:
        certs_path = path.join(path.dirname(__file__))
        client_key = open(f"{certs_path}/client-key.pem", "rb").read()
        client_cert = open(f"{certs_path}/client-cert.pem", "rb").read()
        ca_cert = open(f"{certs_path}/ca-cert.pem", "rb").read()

        creds = grpc.ssl_channel_credentials(
            root_certificates=ca_cert, private_key=client_key, certificate_chain=client_cert
        )
        client = InferenceGRPCClient(url=port,
                                     use_ssl=ssl,
                                     creds=creds,
                                     channel_args=[
                                         # grpc.ssl_target_name_override must be set to match CN used in cert gen
                                         ('grpc.ssl_target_name_override', 'localhost'),]
                                     )
    else:
        client = InferenceGRPCClient(url=port)
    data = float(integer)
    infer_input = InferInput(name="input-0", shape=[1], datatype="FP32", data=[data])
    request = InferRequest(infer_inputs=[infer_input], model_name="custom-model")
    res = await client.infer(infer_request=request)
    return res


@pytest.mark.grpc
@pytest.mark.asyncio(scope="session")
async def test_custom_model_grpc_secure_ssl():
    service_name = "custom-model-grpc"
    model_name = "custom-model"

    predictor = V1beta1PredictorSpec(
        containers=[
            V1Container(
                name="kserve-container",
                image="kserve/custom_model_secure_grpc:" + os.environ.get("GITHUB_SHA"),
                resources=V1ResourceRequirements(
                    requests={"cpu": "50m", "memory": "128Mi"},
                    limits={"cpu": "100m", "memory": "1Gi"},
                ),
                ports=[
                    V1ContainerPort(container_port=8081, name="h2c", protocol="TCP")
                ],
                args=["--model_name", model_name],
            )
        ]
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=service_name, namespace=KSERVE_TEST_NAMESPACE
        ),
        spec=V1beta1InferenceServiceSpec(predictor=predictor),
    )

    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config")
    )
    kserve_client.create(isvc)
    kserve_client.wait_isvc_ready(service_name, namespace=KSERVE_TEST_NAMESPACE)

    response = await grpc_infer_request(1, "localhost:8081", True, [], [])
    number = response.outputs[0].data[0]
    assert number == 2.0
    kserve_client.delete(service_name, KSERVE_TEST_NAMESPACE)
