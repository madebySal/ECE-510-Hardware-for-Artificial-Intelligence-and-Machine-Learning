# Heilmeier Catechism – Project Draft
## Binary Neural Network (BNN) Inference Accelerator

---

## 1. What are you trying to do?

Build a custom hardware chiplet that accelerates inference of Binary Neural Networks (BNNs) — neural networks where weights and activations are constrained to +1/−1. The goal is to replace expensive floating-point multiply-accumulate (MAC) operations with simple XNOR and popcount operations, enabling fast, low-power inference on edge devices without requiring a GPU or general-purpose processor.

---

## 2. How is it done today, and what are the limitations of current practice?

Neural network inference today runs primarily on CPUs or GPUs using FP32 or INT8 arithmetic. GPUs deliver high throughput but consume hundreds of watts and are too large and expensive for edge deployment. CPUs are power-efficient but too slow for real-time inference on non-trivial models. Specialized accelerators (e.g., Google TPU, Apple Neural Engine) exist but are proprietary, closed, and not designed for binary-weight networks. Software BNN libraries such as Larq can run binary networks on standard hardware, but they still use standard integer arithmetic units and cannot exploit the full parallelism that binary representations allow — specifically, the fact that a multiply between two binary values reduces to a single XNOR gate, and accumulation reduces to a popcount.

The key limitations are:
- High power consumption of FP/INT hardware for a task that only needs 1-bit math
- No open, synthesizable BNN accelerator with a standard host interface
- Software implementations leave most of the theoretical efficiency gain on the table

---

## 3. What is new in your approach and why do you think it will be successful?

This project implements a custom XNOR-popcount compute engine in SystemVerilog specifically optimized for BNN inference. The core innovation is replacing multiply-accumulate units with 1-bit XNOR gates and popcount trees — reducing each multiply to a single logic gate and accumulation to bit counting. This yields large reductions in area, power, and latency compared to INT8 or FP16 hardware.

The approach is grounded in published research (XNOR-Net, BinaryConnect, FINN) and the hardware primitives are simple and well-understood, making synthesis with OpenLane 2 realistic within the project timeline. The target application is keyword spotting or gesture recognition — small enough models to verify end-to-end in simulation and benchmark against a CPU software baseline. A standard AXI4-Lite or SPI interface will connect the chiplet to a host, making the design realistic as an engineering artifact.

---

## References
- Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks," ECCV 2016
- Courbariaux et al., "BinaryConnect: Training Deep Neural Networks with Binary Weights," NeurIPS 2015
- Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference," FPGA 2017
