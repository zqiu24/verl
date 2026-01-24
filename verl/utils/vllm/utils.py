# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from msgspec import field
from packaging import version as vs

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel

try:
    from vllm.oft.oft_model import OFTModel
except ImportError:
    from vllm.oft.models import OFTModel

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.oft.request import OFTRequest
from vllm.oft.utils import get_adapter_absolute_path
from vllm.oft.worker_manager import LRUCacheWorkerOFTManager

from verl.third_party.vllm import get_version


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class TensorOFTRequest(OFTRequest):
    peft_config: dict = field(default=None)
    oft_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, request: TensorLoRARequest | TensorOFTRequest) -> LoRAModel | OFTModel:
            """
            Unified adapter loader supporting both LoRA and OFT from memory tensors.
            
            Based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter
            and vllm.oft.worker_manager.WorkerOFTManager._load_adapter
            
            Reason:
            vLLM does not support adding LoRA/OFT from tensors directly. 
            It only supports loading via file paths. To synchronize the adapter 
            tensors from the actor model, we hijack the internal _load_adapter 
            to enable memory-based loading.
            """
            try:
                is_lora = isinstance(request, TensorLoRARequest)
                is_oft = isinstance(request, TensorOFTRequest)
                is_tensor_based = isinstance(request, (TensorLoRARequest, TensorOFTRequest))

                if is_lora:
                    supported_modules = self._adapter_manager.supported_lora_modules
                    config = self.lora_config
                    model_cls = self._lora_model_cls
                    adapter_type = "lora"
                elif is_oft:
                    supported_modules = self._adapter_manager.supported_oft_modules
                    config = self.oft_config
                    model_cls = self._oft_model_cls
                    adapter_type = "oft"
                else:
                    raise ValueError(f"Unsupported adapter type: {type(request)}")

                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_modules: list[str] = []
                for module in supported_modules:
                    if module in packed_modules_mapping:
                        expected_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_modules.append(module)

                expected_modules = list(set(expected_modules))

                if is_lora:
                    from vllm.lora.peft_helper import PEFTHelper
                elif is_oft:
                    from vllm.oft.peft_helper import PEFTHelper
                else:
                    raise ValueError(f"Unsupported adapter type: {type(request)}")

                if is_tensor_based:
                    # Memory-based loading
                    peft_config = request.peft_config
                    adapter_tensors = request.lora_tensors if is_lora else request.oft_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    # File-based loading
                    adapter_path = get_adapter_absolute_path(
                        request.lora_path if is_lora else request.oft_path
                    )
                    peft_helper = PEFTHelper.from_local_dir(adapter_path, self.max_position_embeddings)

                # Validates the LoRA/OFT configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                adapter_request_kwargs = {
                    "peft_helper": peft_helper,
                    "device": "cpu",
                    "dtype": config.lora_dtype if is_lora else config.oft_dtype,
                    "weights_mapper": hf_to_vllm_mapper,
                }

                if is_lora:
                    adapter_request_kwargs["lora_model_id"] = request.lora_int_id
                elif is_oft:
                    adapter_request_kwargs["oft_model_id"] = request.oft_int_id

                if hasattr(self, "embedding_padding_modules"):
                    adapter_request_kwargs["embedding_modules"] = self.embedding_modules
                    adapter_request_kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                else:
                    adapter_request_kwargs["model_vocab_size"] = self.vocab_size

                extra_vocab_attr = "lora_extra_vocab_size" if is_lora else "oft_extra_vocab_size"
                if hasattr(config, extra_vocab_attr):
                    adapter_request_kwargs["target_embedding_padding"] = (
                        self.vocab_size + getattr(config, extra_vocab_attr)
                    )

                if is_tensor_based:
                    load_method = model_cls.from_lora_tensors if is_lora else model_cls.from_oft_tensors
                    # For OFT, pre-compute module dimensions
                    if is_oft:
                        from vllm.model_executor.layers.linear import QKVParallelLinear, MergedColumnParallelLinear
                        # Build a map of module_name -> (input_dim, output_dim)
                        module_dims = {}
                        for name, module in model.named_modules():
                            if hasattr(module, 'weight'):
                                weight_shape = module.weight.shape
                                if len(weight_shape) == 2:
                                    module_dims[name] = {
                                        'output_dim': weight_shape[0],
                                        'input_dim': weight_shape[1]
                                    }

                            # For QKV layers, also add separate q/k/v entries
                            base_module = getattr(module, 'base_layer', module)
                            if isinstance(base_module, QKVParallelLinear):
                                input_dim = base_module.input_size
                                q_output = base_module.total_num_heads * base_module.head_size
                                kv_output = base_module.total_num_kv_heads * base_module.head_size
                                
                                # Add unfused layer dims
                                base_name = name.replace('.base_layer', '')
                                if 'qkv_proj' in base_name:
                                    q_name = base_name.replace('qkv_proj', 'q_proj')
                                    k_name = base_name.replace('qkv_proj', 'k_proj')
                                    v_name = base_name.replace('qkv_proj', 'v_proj')
                                    module_dims[q_name] = {'input_dim': input_dim, 'output_dim': q_output}
                                    module_dims[k_name] = {'input_dim': input_dim, 'output_dim': kv_output}
                                    module_dims[v_name] = {'input_dim': input_dim, 'output_dim': kv_output}

                            # For MLP gate_up_proj layers, add separate gate/up entries
                            if isinstance(base_module, MergedColumnParallelLinear):
                                base_name = name.replace('.base_layer', '')
                                if 'gate_up_proj' in base_name:
                                    input_dim = base_module.input_size
                                    # output_sizes is [gate_size, up_size], typically equal
                                    output_sizes = base_module.output_sizes
                                    gate_name = base_name.replace('gate_up_proj', 'gate_proj')
                                    up_name = base_name.replace('gate_up_proj', 'up_proj')
                                    module_dims[gate_name] = {'input_dim': input_dim, 'output_dim': output_sizes[0]}
                                    module_dims[up_name] = {'input_dim': input_dim, 'output_dim': output_sizes[1]}
                        
                        adapter = load_method(
                            tensors=adapter_tensors,
                            module_dims=module_dims,
                            **adapter_request_kwargs,
                        )
                    else:
                        adapter = load_method(
                            tensors=adapter_tensors,
                            **adapter_request_kwargs,
                        )
                else:
                    adapter_path = model_cls.from_local_checkpoint(
                        adapter_path,
                        expected_modules,
                        **adapter_request_kwargs,
                    )

            except Exception:
                raise

            extra_vocab_attr = "lora_extra_vocab_size" if is_lora else "oft_extra_vocab_size"
            if (getattr(adapter, "extra_vocab_size", 0) > getattr(config, extra_vocab_attr, 0)):
                raise ValueError(
                    f"{adapter_type} added vocab size {adapter.extra_vocab_size} "
                    f"is greater than {extra_vocab_attr} {getattr(config, extra_vocab_attr)}."
                )
            
            return adapter

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)
        do_hijack(LRUCacheWorkerOFTManager, "_load_adapter", hijack__load_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
