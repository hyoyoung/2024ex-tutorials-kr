스키핑 모듈 매개변수 초기화
========================================

소개
------------

모듈이 생성 될 때, 학습 가능한 매개변수들은 모듈 타입과 관련된 기본 초기화 방식에 따라 초기화가 됩니다.

예를 들면, (:class:`torch.nn.Linear`) 모듈의 가중치 매개변수는 `uniform(-1/sqrt(in_features), 1/sqrt(in_features))`로 분배되어 초기화 됩니다.

다른 초기화 방식을 바란다면, 모듈 인스턴스화 이후에 이전의 방식으로 매개변수들을 재 초기화하는 것이 필요합니다.

::

    from torch import nn

    # Initializes weight from the default distribution: uniform(-1/sqrt(10), 1/sqrt(10)).
    m = nn.Linear(10, 5)

    # Re-initialize weight from a different distribution.
    nn.init.orthogonal_(m.weight)

In this case, the initialization done during construction is wasted computation, and it may be non-trivial if
the `weight` parameter is large.

Skipping Initialization
-----------------------

It is now possible to skip parameter initialization during module construction, avoiding
wasted computation. This is easily accomplished using the :func:`torch.nn.utils.skip_init` function:

::

    from torch import nn
    from torch.nn.utils import skip_init

    m = skip_init(nn.Linear, 10, 5)

    # Example: Do custom, non-default parameter initialization.
    nn.init.orthogonal_(m.weight)

This can be applied to any module that satisfies the conditions described in the
:ref:`Updating` section below. Note that all modules provided by
`torch.nn` satisfy these conditions and thus support skipping init.

.. _Updating:

Updating Modules to Support Skipping Initialization
---------------------------------------------------

Due to the way :func:`torch.nn.utils.skip_init` is implemented (see :ref:`Details`), there are
two requirements that a module must meet to be compatible with the function.
You can opt in to the parameter initialization skipping functionality for your custom module
simply by adhering to these requirements:

  1. The module must accept a `device` kwarg in its constructor that is passed to any parameters
  or buffers created during construction.

  2. The module must not perform any computation on parameters or buffers in its constructor except
  initialization (i.e. functions from `torch.nn.init`).

The following example demonstrates a module updated to support the `device`
kwarg by passing it along to any created parameters, buffers, or submodules:

::

    import torch
    from torch import nn

    class MyModule(torch.nn.Module):
      def __init__(self, foo, bar, device=None):
        super().__init__()

        # ==== Case 1: Module creates parameters directly. ====
        # Pass device along to any created parameters.
        self.param1 = nn.Parameter(torch.empty((foo, bar), device=device))
        self.register_parameter('param2', nn.Parameter(torch.empty(bar, device=device)))

        # To ensure support for the meta device, avoid using ops except those in
        # torch.nn.init on parameters in your module's constructor.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.param1)
            nn.init.uniform_(self.param2)


        # ==== Case 2: Module creates submodules. ====
        # Pass device along recursively. All submodules will need to support
        # them as well; this is the case for all torch.nn provided modules.
        self.fc = nn.Linear(bar, 5, device=device)

        # This also works with containers.
        self.linears = nn.Sequential(
            nn.Linear(5, 5, device=device),
            nn.Linear(5, 1, device=device)
        )


        # ==== Case 3: Module creates buffers. ====
        # Pass device along during buffer tensor creation.
        self.register_buffer('some_buffer', torch.ones(7, device=device))

    ...

.. _Details:

Implementation Details
----------------------

Behind the scenes, the :func:`torch.nn.utils.skip_init` function is implemented in terms of a two-step pattern:

::

    # 1. Initialize module on the meta device; all torch.nn.init ops have
    # no-op behavior on the meta device.
    m = nn.Linear(10, 5, device='meta')

    # 2. Materialize an uninitialized (empty) form of the module on the CPU device.
    # The result of this is a module instance with uninitialized parameters.
    m.to_empty(device='cpu')

It works by instantiating the module onto a "meta" device, which has tensor shape information
but does not allocate any storage. The `torch.nn.init` ops are specially implemented for this meta device
so that they have no-op behavior. This results in the parameter intialization logic being essentially skipped.

Note that this pattern only works for modules that properly support a `device` kwarg during construction, as
described in :ref:`Updating`.
