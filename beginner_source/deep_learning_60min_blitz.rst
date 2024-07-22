PyTorch로 딥 러닝 60분 만에 뽀개기 
---------------------------------------------
**저자**: `Soumith Chintala <http://soumith.ch>`_

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/u7x8RXwLKcA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

PyTorch란 무엇인가요?
~~~~~~~~~~~~~~~~~~~~~
PyTorch는 파이썬 기반 과학 컴퓨팅 패키지로, 두 가지 목적을 갖습니다:

-  GPU나 다른 가속기의 힘을 사용하기 위한 NumPy의 대체제 제공
-  신경망 구현에 유용한 자동 차별화 라이브러리 제공

이 튜토리얼의 목표:
~~~~~~~~~~~~~~~~~~~~~~~~
- PyTorch의 tensor 라이브러리와 신경망에 대한 높은 이해도를 갖습니다.
- 이미지 분류를 위한 소규모 신경망을 훈련합니다.

튜토리얼을 시작하기 전에, `torch`_, `torchvision`_,
및 `matplotlib`_ 패키지가 설치되어 있는지 확인해 주세요.

.. _torch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision
.. _matplotlib: https://github.com/matplotlib/matplotlib

.. toctree::
   :hidden:

   /beginner/blitz/tensor_tutorial
   /beginner/blitz/autograd_tutorial
   /beginner/blitz/neural_networks_tutorial
   /beginner/blitz/cifar10_tutorial

.. grid:: 4

   .. grid-item-card::  :octicon:`file-code;1em` Tensors
      :link: blitz/tensor_tutorial.html

      In this tutorial, you will learn the basics of PyTorch tensors.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` A Gentle Introduction to torch.autograd
      :link: blitz/autograd_tutorial.html

      Learn about autograd.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Neural Networks
      :link: blitz/neural_networks_tutorial.html

      This tutorial demonstrates how you can train neural networks in PyTorch.
      +++
      :octicon:`code;1em` Code

   .. grid-item-card::  :octicon:`file-code;1em` Training a Classifier
      :link: blitz/cifar10_tutorial.html

      Learn how to train an image classifier in PyTorch by using the
      CIFAR10 dataset.
      +++
      :octicon:`code;1em` Code

