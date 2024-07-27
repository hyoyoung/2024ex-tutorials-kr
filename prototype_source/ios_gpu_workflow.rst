(Prototype) PyTorch에서 iOS GPU에 관한 사용 설명서
==================================

**저자**: `Tao Xu <https://github.com/xta0>`_

소개
------------

이 설명서는 iOS GPU에서 모델을 실행하는 방법에 대해 소개합니다. 우리는 mobilenetv2 모델을 예시로 사용할 것입니다. 모바일에서 GPU 기능들은 현재 프로토타입 단계에 있으므로 소스에서 직접 PyTorch 바이너리를 커스텀해서 빌드해야 합니다. 당분간은 제한된 수의 연산자만 지원되며, 특정 클라이언트 API는 향후 버전에서 변경 될 가능성이 있습니다.

모델 준비
-------------------

GPU들은 가중치를 다른 순서로 소비하므로 가장 먼저 해야 할 일은 TorchScript 모델을 GPU 호환 모델로 변환하는 것입니다. 이 단계를 "prepacking"이라고도 합니다.

PyTorch와 Metal
^^^^^^^^^^^^^^^^^^
그렇게 하려면, Metal 백엔드를 포함하는 PyTorch Nightly 바이너리를 설치해야 합니다. 다음과 같이 명령을 실행하세요.

.. code:: shell

    conda install pytorch -c pytorch-nightly
    // or
    pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

또한, Metal 백엔드를 포함하는 소스에서 커스텀 PyTorch 바이너리를 빌드할 수 있습니다. github에서 PyTorch를 checkout하고 다음과 같이 명령을 실행하세요.

.. code:: shell

    cd PYTORCH_ROOT
    USE_PYTORCH_METAL_EXPORT=ON python setup.py install --cmake

위의 명령은 master에서 커스텀 PyTorch 바이너리를 빌드합니다. ``install`` 인자는 단순히 말해서 ``setup.py``에 기존 PyTorch를 덮어쓰는 것입니다. 빌드가 완료되면, 다른 터미널을 열어 PyTorch 버전을 확인하여 설치가 성공했는지 확인합니다. 이 레시피를 작성할 당시 버전은 ``1.8.0a0+41237a4`` 입니다. master에서 코드를 체크아웃하는 시점에 따라 다른 숫자를 볼 수 있지만, 1.7.0보다 상위 버전이어야 합니다.

.. code:: python

    import torch
    torch.__version__ #1.8.0a0+41237a4

Metal Compatible Model
^^^^^^^^^^^^^^^^^^^^^^

The next step is going to be converting the mobilenetv2 torchscript model to a Metal compatible model. We'll be leveraging the ``optimize_for_mobile`` API from the ``torch.utils`` module. As shown below

.. code:: python

    import torch
    import torchvision
    from torch.utils.mobile_optimizer import optimize_for_mobile

    model = torchvision.models.mobilenet_v2(pretrained=True)
    scripted_model = torch.jit.script(model)
    optimized_model = optimize_for_mobile(scripted_model, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter('./mobilenetv2_metal.pt')

Note that the ``torch.jit.export_opnames(optimized_model)`` is going to dump all the optimized operators from the ``optimized_mobile``. If everything works well, you should be able to see the following ops being printed out from the console


.. code:: shell

    ['aten::adaptive_avg_pool2d',
    'aten::add.Tensor',
    'aten::addmm',
    'aten::reshape',
    'aten::size.int',
    'metal::copy_to_host',
    'metal_prepack::conv2d_run']

Those are all the ops we need to run the mobilenetv2 model on iOS GPU. Cool! Now that you have the ``mobilenetv2_metal.pt`` saved on your disk, let's move on to the iOS part.


Use PyTorch iOS library with Metal
----------------------------------
The PyTorch iOS library with Metal support ``LibTorch-Lite-Nightly`` is available in Cocoapods. You can read the `Using the Nightly PyTorch iOS Libraries in CocoaPods <https://pytorch.org/mobile/ios/#using-the-nightly-pytorch-ios-libraries-in-cocoapods>`_ section from the iOS tutorial for more detail about its usage. 

We also have the `HelloWorld-Metal example <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld-Metal>`_ that shows how to conect all pieces together.  

Note that if you run the HelloWorld-Metal example, you may notice that the results are slighly different from the `results <https://pytorch.org/mobile/ios/#install-libtorch-via-cocoapods>`_ we got from the CPU model as shown in the iOS tutorial.

.. code:: shell

    - timber wolf, grey wolf, gray wolf, Canis lupus
    - malamute, malemute, Alaskan malamute
    - Eskimo dog, husky

This is because by default Metal uses fp16 rather than fp32 to compute. The precision loss is expected. 


Use LibTorch-Lite Built from Source
-----------------------------------

You can also build a custom LibTorch-Lite from Source and use it to run GPU models on iOS Metal. In this section, we'll be using the `HelloWorld example <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ to demonstrate this process. 

First, make sure you have deleted the **build** folder from the "Model Preparation" step in PyTorch root directory. Then run the command below

.. code:: shell

    IOS_ARCH=arm64 USE_PYTORCH_METAL=1 ./scripts/build_ios.sh

Note ``IOS_ARCH`` tells the script to build a arm64 version of Libtorch-Lite. This is because in PyTorch, Metal is only available for the iOS devices that support the Apple A9 chip or above. Once the build finished, follow the `Build PyTorch iOS libraries from source <https://pytorch.org/mobile/ios/#build-pytorch-ios-libraries-from-source>`_ section from the iOS tutorial to setup the XCode settings properly. Don't forget to copy the ``./mobilenetv2_metal.pt`` to your XCode project and modify the model file path accordingly.

Next we need to make some changes in ``TorchModule.mm``

.. code:: objective-c

    ...
    // #import <Libtorch-Lite/Libtorch-Lite.h>
    // If it's built from source with Xcode, comment out the line above
    // and use following headers
    #include <torch/csrc/jit/mobile/import.h>
    #include <torch/csrc/jit/mobile/module.h>
    #include <torch/script.h>
    ...

    - (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
      c10::InferenceMode mode;
      at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat).metal();
      auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
      ...
    }
    ...

As you can see, we simply just call ``.metal()`` to move our input tensor from CPU to GPU, and then call ``.cpu()`` to move the result back. Internally, ``.metal()`` will copy the input data from the CPU buffer to a GPU buffer with a GPU compatible memory format. When ``.cpu()`` is invoked, the GPU command buffer will be flushed and synced. After `forward` finished, the final result will then be copied back from the GPU buffer back to a CPU buffer.

The last step we have to do is to add the ``Accelerate.framework`` and the ``MetalPerformanceShaders.framework`` to your xcode project (Open your project via XCode, go to your project target’s "General" tab, locate the "Frameworks, Libraries and Embedded Content" section and click the "+" button).

If everything works fine, you should be able to see the inference results on your phone. 


Conclusion
----------

In this tutorial, we demonstrated how to convert a mobilenetv2 model to a GPU compatible model. We walked through a HelloWorld example to show how to use the C++ APIs to run models on iOS GPU. Please be aware of that GPU feature is still under development, new operators will continue to be added. APIs are subject to change in the future versions.

Thanks for reading! As always, we welcome any feedback, so please create an issue `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.

Learn More
----------

- The `Mobilenetv2 <https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>`_ from Torchvision
- To learn more about how to use ``optimize_for_mobile``, please refer to the `Mobile Perf Recipe <https://pytorch.org/tutorials/recipes/mobile_perf.html>`_
