TorchRec 소개
========================

.. tip::
   본 튜토리얼을 최대한 활용하려면, 이
   `Colab 버전 <https://colab.research.google.com/github/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb>`__ 을 사용하는 것을 권장합니다.
   이를 통해 아래에 제시된 정보를 실험해볼 수 있습니다.
   
아래 비디오 또는 `유튜브 영상 <https://www.youtube.com/watch?v=cjgj41dvSeQ>`__ 을 시청하면서 따라해보세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/cjgj41dvSeQ" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

추천 시스템을 구축할 때, 상품이나 페이지와 같은 개체(Entity)를 임베딩으로 표현해야 하는 경우가 많습니다.  
예를 들면, Meta
AI의 `딥 러닝 추천 모델 <https://arxiv.org/abs/1906.00091>`__ (Deep learning recommendation model 또는 DLRM)과 같은 경우입니다.
개체 수가 증가함에 따라 임베딩 테이블들의 크기가 단일 GPU의 메모리를 초과할 수 있습니다.
이럴 때 흔히 사용하는 방법은 임베딩 테이블을 여러 장치에 샤딩(shard)하는 모델 병렬화입니다.
이를 위해 TorchRec은 주요 API인 |DistributedModelParallel|_ (DMP)을 도입했습니다.
PyTorch의 DistributedDataParallel 처럼, DMP는 분산학습을 할수 있도록 모델을 감쌉(wrap)니다.

Installation
------------

Requirements: python >= 3.7

We highly recommend CUDA when using TorchRec (If using CUDA: cuda >= 11.0).


.. code:: shell

    # install pytorch with cudatoolkit 11.3
    conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y
    # install TorchRec
    pip3 install torchrec-nightly


Overview
--------

This tutorial will cover three pieces of TorchRec: the ``nn.module`` |EmbeddingBagCollection|_, the |DistributedModelParallel|_ API, and
the datastructure |KeyedJaggedTensor|_.


Distributed Setup
~~~~~~~~~~~~~~~~~

We setup our environment with torch.distributed. For more info on
distributed, see this
`tutorial <https://pytorch.org/tutorials/beginner/dist_overview.html>`__.

Here, we use one rank (the colab process) corresponding to our 1 colab
GPU.

.. code:: python

    import os
    import torch
    import torchrec
    import torch.distributed as dist

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Note - you will need a V100 or A100 to run tutorial as as!
    # If using an older GPU (such as colab free K80), 
    # you will need to compile fbgemm with the appripriate CUDA architecture
    # or run with "gloo" on CPUs 
    dist.init_process_group(backend="nccl")


From EmbeddingBag to EmbeddingBagCollection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch represents embeddings through |torch.nn.Embedding|_ and |torch.nn.EmbeddingBag|_.
EmbeddingBag is a pooled version of Embedding.

TorchRec extends these modules by creating collections of embeddings. We
will use |EmbeddingBagCollection|_ to represent a group of EmbeddingBags.

Here, we create an EmbeddingBagCollection (EBC) with two embedding bags.
Each table, ``product_table`` and ``user_table``, is represented by a 64
dimension embedding of size 4096. Note how we initially allocate the EBC
on device “meta”. This will tell EBC to not allocate memory yet.

.. code:: python

    ebc = torchrec.EmbeddingBagCollection(
        device="meta",
        tables=[
            torchrec.EmbeddingBagConfig(
                name="product_table",
                embedding_dim=64,
                num_embeddings=4096,
                feature_names=["product"],
                pooling=torchrec.PoolingType.SUM,
            ),
            torchrec.EmbeddingBagConfig(
                name="user_table",
                embedding_dim=64,
                num_embeddings=4096,
                feature_names=["user"],
                pooling=torchrec.PoolingType.SUM,
            )
        ]
    )


DistributedModelParallel
~~~~~~~~~~~~~~~~~~~~~~~~

Now, we’re ready to wrap our model with |DistributedModelParallel|_ (DMP). Instantiating DMP will:

1. Decide how to shard the model. DMP will collect the available
   ‘sharders’ and come up with a ‘plan’ of the optimal way to shard the
   embedding table(s) (i.e., the EmbeddingBagCollection).
2. Actually shard the model. This includes allocating memory for each
   embedding table on the appropriate device(s).

In this toy example, since we have two EmbeddingTables and one GPU,
TorchRec will place both on the single GPU.

.. code:: python

    model = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
    print(model)
    print(model.plan)


Query vanilla nn.EmbeddingBag with input and offsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We query |nn.Embedding|_ and |nn.EmbeddingBag|_
with ``input`` and ``offsets``. Input is a 1-D tensor containing the
lookup values. Offsets is a 1-D tensor where the sequence is a
cumulative sum of the number of values to pool per example.

Let’s look at an example, recreating the product EmbeddingBag above:

::

   |------------|
   | product ID |
   |------------|
   | [101, 202] |
   | []         |
   | [303]      |
   |------------|

.. code:: python

    product_eb = torch.nn.EmbeddingBag(4096, 64)
    product_eb(input=torch.tensor([101, 202, 303]), offsets=torch.tensor([0, 2, 2]))


Representing minibatches with KeyedJaggedTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need an efficient representation of multiple examples of an arbitrary
number of entity IDs per feature per example. In order to enable this
“jagged” representation, we use the TorchRec datastructure
|KeyedJaggedTensor|_ (KJT).

Let’s take a look at how to lookup a collection of two embedding
bags, “product” and “user”. Assume the minibatch is made up of three
examples for three users. The first of which has two product IDs, the
second with none, and the third with one product ID.

::

   |------------|------------|
   | product ID | user ID    |
   |------------|------------|
   | [101, 202] | [404]      |
   | []         | [505]      |
   | [303]      | [606]      |
   |------------|------------|

The query should be:

.. code:: python

    mb = torchrec.KeyedJaggedTensor(
        keys = ["product", "user"],
        values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
        lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
    )

    print(mb.to(torch.device("cpu")))


Note that the KJT batch size is
``batch_size = len(lengths)//len(keys)``. In the above example,
batch_size is 3.



Putting it all together, querying our distributed model with a KJT minibatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we can query our model using our minibatch of products and
users.

The resulting lookup will contain a KeyedTensor, where each key (or
feature) contains a 2D tensor of size 3x64 (batch_size x embedding_dim).

.. code:: python

    pooled_embeddings = model(mb)
    print(pooled_embeddings)


More resources
--------------

For more information, please see our
`dlrm <https://github.com/pytorch/torchrec/tree/main/examples/dlrm>`__
example, which includes multinode training on the criteo terabyte
dataset, using Meta’s `DLRM <https://arxiv.org/abs/1906.00091>`__.


.. |DistributedModelParallel| replace:: ``DistributedModelParallel``
.. _DistributedModelParallel: https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.model_parallel.DistributedModelParallel
.. |EmbeddingBagCollection| replace:: ``EmbeddingBagCollection``
.. _EmbeddingBagCollection: https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollection
.. |KeyedJaggedTensor| replace:: ``KeyedJaggedTensor``
.. _KeyedJaggedTensor: https://pytorch.org/torchrec/torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor
.. |torch.nn.Embedding| replace:: ``torch.nn.Embedding``
.. _torch.nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
.. |torch.nn.EmbeddingBag| replace:: ``torch.nn.EmbeddingBag``
.. _torch.nn.EmbeddingBag: https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
.. |nn.Embedding| replace:: ``nn.Embedding``
.. _nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
.. |nn.EmbeddingBag| replace:: ``nn.EmbeddingBag``
.. _nn.EmbeddingBag: https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
