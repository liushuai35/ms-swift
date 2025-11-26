# model, processor(tokenizer), etc.
# get_function
# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from typing import Any, Dict, Type

import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_device_count, get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_get_input_embeddings, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl, ModelInfo, safe_snapshot_download

logger = get_logger()


def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase], tokenizer_config: Dict[str, Any]) -> None:
    for k, v in tokenizer_cls.__dict__.items():
        if k.endswith('_token') and isinstance(v, property) and k in tokenizer_config:
            setattr(tokenizer_cls, k, tokenizer_config[k])


def _patch_tokenizer(tokenizer):
    tokenizer_cls = tokenizer.__class__
    if hasattr(tokenizer_cls, '_origin_pad'):
        return
    tokenizer_cls._origin_pad = tokenizer_cls._pad
    parameters = inspect.signature(tokenizer_cls._origin_pad).parameters

    def _pad(self, *args, **kwargs):
        if 'padding_side' in kwargs and kwargs['padding_side'] is None and 'padding_side' not in parameters:
            kwargs.pop('padding_side')
        return tokenizer_cls._origin_pad(self, *args, **kwargs)

    tokenizer_cls._pad = _pad


def get_model_tokenizer_custom(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    _patch_tokenizer(tokenizer)
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.local_deepseek, [
            ModelGroup([
                Model(ms_model_id='local/deepseek-base', model_path='local/deepseek-base'),
            ])
        ],
        TemplateType.deepseek_v3_1,
        get_model_tokenizer_custom,
        tags=['local_deepseek']))