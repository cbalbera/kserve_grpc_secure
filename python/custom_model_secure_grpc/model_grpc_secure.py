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
import io
from typing import Dict

from kserve import InferRequest, Model, ModelServer, logging, model_server, InferResponse, InferOutput
from kserve.utils.utils import generate_uuid
from os import path


# This custom predictor example implements the custom model following KServe v2 inference gPPC protocol,
# the input can be raw image bytes or image tensor which is pre-processed by transformer
# and then passed to predictor, the output is the prediction response.
class TestModelForSecure(Model):  # Test model
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.load()

    def load(self):
        self.ready = True

    # Returns a number + 1
    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> InferResponse:
        req = payload.inputs[0]
        input_number = req.data[0]  # Input should be a single number
        assert isinstance(input_number, (int, float)), "Data is not a number or float"
        result = [float(input_number + 1)]

        response_id = generate_uuid()
        infer_output = InferOutput(name="output-0", shape=[1], datatype="FP32", data=result)
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        return infer_response


if __name__ == "__main__":
    model = TestModelForSecure("custom-model")
    certs_path = path.join(path.dirname(__file__), "kserve_test_certs")
    server_key = open(f"{certs_path}/server-key.pem", 'rb').read()
    server_cert = open(f"{certs_path}/server-cert.pem", 'rb').read()
    ca_cert = open(f"{certs_path}/ca-cert.pem", 'rb').read()
    ModelServer(
        secure_grpc_server=True,
        ssl_server_key=server_key,
        ssl_server_cert=server_cert,
        ssl_ca_cert=ca_cert
    ).start([model])
