**Introduction** \|\| `What is DDP <ddp_series_theory.html>`__ \|\|
`Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ \|\|
`Fault Tolerance <ddp_series_fault_tolerance.html>`__ \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__

파이토치에서 분산 데이터 병렬 처리하기 - 비디오 튜토리얼
======================================================

저자: `Suraj Subramanian <https://github.com/suraj813>`__
번역: `박지은 <https://github.com/rumjie>`__
아래 혹은 링크의 영상과 함께 따라해보세요. `youtube <https://www.youtube.com/watch/-K3bZYHYHEA>`__

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/-K3bZYHYHEA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

이 영상 튜토리얼들은 분산 데이터 병렬 처리를 통해 파이토치에서 분산 학습을 시작할 수 있도록 돕습니다.

이 영상들은 간단한 비분산 학습 작업부터 시작해 
클러스터 내 여러 대의 머신들에 학습 작업을 배포하는 것으로 마무리됩니다. 
이 과정에서, 여러분은 결함 허용 분산 학습을 위한 
`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__ 에 대해서도 배울 수 있습니다.

The tutorial assumes a basic familiarity with model training in PyTorch.

Running the code
----------------

You will need multiple CUDA GPUs to run the tutorial code. Typically,
this can be done on a cloud instance with multiple GPUs (the tutorials
use an Amazon EC2 P3 instance with 4 GPUs).

The tutorial code is hosted in this
`github repo <https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series>`__.
Clone the repository and follow along!

Tutorial sections
-----------------

0. Introduction (this page)
1. `What is DDP? <ddp_series_theory.html>`__ Gently introduces what DDP is doing
   under the hood
2. `Single-Node Multi-GPU Training <ddp_series_multigpu.html>`__ Training models
   using multiple GPUs on a single machine
3. `Fault-tolerant distributed training <ddp_series_fault_tolerance.html>`__
   Making your distributed training job robust with torchrun
4. `Multi-Node training <../intermediate/ddp_series_multinode.html>`__ Training models using
   multiple GPUs on multiple machines
5. `Training a GPT model with DDP <../intermediate/ddp_series_minGPT.html>`__ “Real-world”
   example of training a `minGPT <https://github.com/karpathy/minGPT>`__
   model with DDP
