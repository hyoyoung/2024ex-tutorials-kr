(prototype) PyTorch BackendConfig 튜토리얼
==========================================
**저자**: `Andrew Or <https://github.com/andrewor14>`_

BackendConfig API는 개발자가 자신들의 백엔드를 PyTorch 양자화와 통합할 수 있도록 해줍니다. 현재는 FX 그래프 모드 양자화에서만 지원되지만, 앞으로는 다른 양자화 모드에도 지원이 확장될 수 있습니다. 
이 튜토리얼에서는 이 API를 사용하여 특정 백엔드에 대한 양자화 지원을 사용자 정의하는 방법을 시연합니다. BackendConfig의 동기와 구현 세부사항에 대한 자세한 정보는 이 README를 참조하십시오.
`README <https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config>`__.

우리가 백엔드 개발자라고 가정하고, PyTorch의 양자화 API와 우리의 백엔드를 통합하고자 한다고 가정해 보겠습니다. 우리의 백엔드는 양자화된 linear와 양자화된 conv-relu 두 가지 연산만으로 구성되어 있습니다. 이 섹션에서는 `prepare_fx`와 `convert_fx`를 사용하여 사용자 정의 BackendConfig 를 통해 예제 모델을 양자화하는 방법을 단계별로 설명하겠습니다. 

.. code:: ipython3

    import torch
    from torch.ao.quantization import (
        default_weight_observer,
        get_default_qconfig_mapping,
        MinMaxObserver,
        QConfig,
        QConfigMapping,
    )
    from torch.ao.quantization.backend_config import (
        BackendConfig,
        BackendPatternConfig,
        DTypeConfig,
        DTypeWithConstraints,
        ObservationType,
    )
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

1. 각 양자화 연산자에 대한 참조 패턴 도출
--------------------------------------------------------
양자화된 linear의 경우, 우리의 백엔드가 `[dequant - fp32_linear - quant]` 참조 패턴을 예상하고 이를 단일 양자화된 linear 연산으로 낮춘다고 가정해 보겠습니다. 
이를 달성하기 위한 방법은 먼저 float linear 연산 전후에 quant-dequant 연산을 삽입하여 다음과 같은 참조 모델을 생성하는 것입니다.

  quant1 - [dequant1 - fp32_linear - quant2] - dequant2


마찬가지로, 양자화된 conv-relu의 경우, 다음과 같은 참조 모델을 생성하려고 합니다. 여기서 대괄호 안의 참조 패턴은 단일 양자화된 conv-relu 연산으로 낮춰질 것입니다:

  quant1 - [dequant1 - fp32_conv_relu - quant2] - dequant2

2. DTypeConfigs에 백엔드 제약 조건 설정
---------------------------------------------

위 참조 패턴에서 DTypeConfig에 지정된 입력 dtype은 quant1에 dtype 인수로 전달되고, 출력 dtype은 quant2에 dtype 인수로 전달됩니다. 
만약 출력 dtype이 동적 양자화의 경우처럼 fp32이면, 출력 quant-dequant 쌍이 삽입되지 않습니다. 
이 예제는 특정 dtype에 대한 양자화 및 스케일 범위에 대한 제약 조건을 지정하는 방법도 보여줍니다.

.. code:: ipython3

    quint8_with_constraints = DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 ** -12,
    )
    
    # Specify the dtypes passed to the quantized ops in the reference model spec
    weighted_int8_dtype_config = DTypeConfig(
        input_dtype=quint8_with_constraints,
        output_dtype=quint8_with_constraints,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float)

3. conv-relu에 대한 결합 설정
-------------------------------

원래의 사용자 모델은 개별적인 conv와 relu 연산을 포함하고 있으므로, 먼저 conv와 relu 연산을 단일 conv-relu 연산(`fp32_conv_relu`)으로 결합한 후, 이 연산을 linear 연산을 양자화하는 방식과 유사하게 양자화해야 합니다. 
결합을 설정하려면 QAT인지 여부와 결합된 패턴의 개별 항목을 나타내는 두 가지 인수를 받아들이는 함수를 정의하면 됩니다.

.. code:: ipython3

   def fuse_conv2d_relu(is_qat, conv, relu):
       """Return a fused ConvReLU2d from individual conv and relu modules."""
       return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

4. BackendConfig 정의
----------------------------

이제 필요한 모든 요소가 준비되었으므로, BackendConfig를 정의할 수 있습니다. 여기서는 linear 연산의 입력과 출력에 대해 서로 다른 옵저버를 사용합니다. 
이렇게 하면 두 개의 양자화 연산(quant1과 quant2)에 전달되는 양자화 파라미터가 다르게 됩니다. 
이는 일반적으로 linear와 conv 같은 가중치를 가진 연산에서 흔히 발생합니다.
conv-relu 연산의 경우, 관측 유형이 동일합니다. 그러나 이 연산을 지원하기 위해 결합용 하나와 양자화용 하나, 두 개의 BackendPatternConfig가 필요합니다. 
conv-relu와 linear 모두에 대해 위에서 정의한 DTypeConfig를 사용합니다.

.. code:: ipython3

    linear_config = BackendPatternConfig() \
        .set_pattern(torch.nn.Linear) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .add_dtype_config(weighted_int8_dtype_config) \
        .set_root_module(torch.nn.Linear) \
        .set_qat_module(torch.nn.qat.Linear) \
        .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

    # For fusing Conv2d + ReLU into ConvReLU2d
    # No need to set observation type and dtype config here, since we are not
    # inserting quant-dequant ops in this step yet
    conv_relu_config = BackendPatternConfig() \
        .set_pattern((torch.nn.Conv2d, torch.nn.ReLU)) \
        .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
        .set_fuser_method(fuse_conv2d_relu)
    
    # For quantizing ConvReLU2d
    fused_conv_relu_config = BackendPatternConfig() \
        .set_pattern(torch.ao.nn.intrinsic.ConvReLU2d) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .add_dtype_config(weighted_int8_dtype_config) \
        .set_root_module(torch.nn.Conv2d) \
        .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
        .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

    backend_config = BackendConfig("my_backend") \
        .set_backend_pattern_config(linear_config) \
        .set_backend_pattern_config(conv_relu_config) \
        .set_backend_pattern_config(fused_conv_relu_config)

5. 백엔드 제약 조건을 만족하는 QConfigMapping 설정

----------------------------------------------------------------
위에서 정의한 연산을 사용하려면 사용자는 DTypeConfig에 지정된 제약 조건을 만족하는 QConfig를 정의해야 합니다. 자세한 내용은 `DTypeConfig <https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.DTypeConfig.html>`__. 문서를 참조하십시오. 그런 다음 이 QConfig를 우리가 양자화하려는 패턴에 사용된 모든 모듈에 사용할 것입니다.




.. code:: ipython3

    # Note: Here we use a quant_max of 127, but this could be up to 255 (see `quint8_with_constraints`)
    activation_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127, eps=2 ** -12)
    qconfig = QConfig(activation=activation_observer, weight=default_weight_observer)

    # Note: All individual items of a fused pattern, e.g. Conv2d and ReLU in
    # (Conv2d, ReLU), must have the same QConfig
    qconfig_mapping = QConfigMapping() \
        .set_object_type(torch.nn.Linear, qconfig) \
        .set_object_type(torch.nn.Conv2d, qconfig) \
        .set_object_type(torch.nn.BatchNorm2d, qconfig) \
        .set_object_type(torch.nn.ReLU, qconfig)

6. 준비 및 변환을 통해 모델 양자화
--------------------------------------------------

마지막으로, 우리가 정의한 BackendConfig를 준비(prepare) 및 변환(convert) 단계에 전달하여 모델을 양자화합니다. 이 과정에서 양자화된 linear 모듈과 결합된 양자화된 conv-relu 모듈이 생성됩니다.

.. code:: ipython3

    class MyModel(torch.nn.Module):
        def __init__(self, use_bn: bool):
            super().__init__()
            self.linear = torch.nn.Linear(10, 3)
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.bn = torch.nn.BatchNorm2d(3)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.use_bn = use_bn

        def forward(self, x):
            x = self.linear(x)
            x = self.conv(x)
            if self.use_bn:
                x = self.bn(x)
            x = self.relu(x)
            x = self.sigmoid(x)
            return x

    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=False)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.012136868201196194, zero_point=67, qscheme=torch.per_tensor_affine)
      (conv): QuantizedConvReLU2d(3, 3, kernel_size=(3, 3), stride=(1, 1), scale=0.0029353597201406956, zero_point=0)
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear_input_scale_0 = self.linear_input_scale_0
        linear_input_zero_point_0 = self.linear_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
        linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
        conv = self.conv(linear);  linear = None
        dequantize_2 = conv.dequantize();  conv = None
        sigmoid = self.sigmoid(dequantize_2);  dequantize_2 = None
        return sigmoid

(7. 결함이 있는 BackendConfig 설정으로 실험하기)
-------------------------------------------------

실험으로, 모델에서 conv-relu 대신 conv-bn-relu를 사용하도록 수정하지만, conv-bn-relu를 양자화하는 방법을 모르는 동일한 BackendConfig를 사용합니다. 
그 결과, linear만 양자화되고 conv-bn-relu는 결합되지도 양자화되지도 않습니다.

.. code:: ipython3
    # Only linear is quantized, since there's no rule for fusing conv-bn-relu
    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=True)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): QuantizedLinear(in_features=10, out_features=3, scale=0.015307803638279438, zero_point=95, qscheme=torch.per_tensor_affine)
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear_input_scale_0 = self.linear_input_scale_0
        linear_input_zero_point_0 = self.linear_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
        linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
        dequantize_1 = linear.dequantize();  linear = None
        conv = self.conv(dequantize_1);  dequantize_1 = None
        bn = self.bn(conv);  conv = None
        relu = self.relu(bn);  bn = None
        sigmoid = self.sigmoid(relu);  relu = None
        return sigmoid

또 다른 실험으로, 백엔드에서 지정한 dtype 제약 조건을 만족하지 않는 기본 QConfigMapping을 사용합니다. 
그 결과, QConfig가 무시되어 아무것도 양자화되지 않습니다.

.. code:: ipython3
    # Nothing is quantized or fused, since backend constraints are not satisfied
    example_inputs = (torch.rand(1, 3, 10, 10, dtype=torch.float),)
    model = MyModel(use_bn=True)
    prepared = prepare_fx(model, get_default_qconfig_mapping(), example_inputs, backend_config=backend_config)
    prepared(*example_inputs)  # calibrate
    converted = convert_fx(prepared, backend_config=backend_config)

.. parsed-literal::

    >>> print(converted)

    GraphModule(
      (linear): Linear(in_features=10, out_features=3, bias=True)
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (sigmoid): Sigmoid()
    )
    
    def forward(self, x):
        linear = self.linear(x);  x = None
        conv = self.conv(linear);  linear = None
        bn = self.bn(conv);  conv = None
        relu = self.relu(bn);  bn = None
        sigmoid = self.sigmoid(relu);  relu = None
        return sigmoid


내장 BackendConfig
-----------------------

PyTorch 양자화는 ``torch.ao.quantization.backend_config``네임스페이스에서 몇 가지 기본 내장 BackendConfig를 지원합니다:

- `get_fbgemm_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/fbgemm.py>`__:
  서버 대상 설정을 위한 설정
- `get_qnnpack_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/qnnpack.py>`__:
  모바일 및 엣지 디바이스 대상 설정을 위한 설정, XNNPACK 양자화 연산도 지원
- `get_native_backend_config <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/native.py>`__
  (기본값): FBGEMM 및 QNNPACK BackendConfig에서 지원하는 연산자 패턴의 결합을 지원하는 BackendConfig

또한 TensorRT와 x86과 같은 다른 BackendConfig도 개발 중이지만, 현재는 대부분 실험적인 단계에 있습니다. 
사용자가 PyTorch의 양자화 API와 새로운 맞춤형 백엔드를 통합하려는 경우, 위의 예제에서 사용된 것과 동일한 API 세트를 사용하여 자체 BackendConfig를 정의할 수 있습니다.

추가 읽기
---------------

FX 그래프 모드 양자화에서 BackendConfig가 사용되는 방법:
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/README.md

BackendConfig의 동기와 구현 세부사항:
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

BackendConfig의 초기 설계:
https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md
