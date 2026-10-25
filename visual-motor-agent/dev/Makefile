# Makefile for the visual-motor agent (C++ model, Python world)
#
# CPU build: g++ + OpenMP + SSE intrinsics. -ffp-contract=off keeps acting (KV cache, M=1 GEMMs) and
# training (M=T GEMMs) bitwise identical. No GPU is required (libvmcuda.so, the optional GPU update, needs one).

CXX = g++
CXXFLAGS = -O3 -march=native -ffp-contract=off -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-parameter
PY = ./venv/bin/python

all: libvmagent.so

# Built under a temporary name and renamed: running trainers keep their mapped copy intact.
libvmagent.so: src/vm_agent.cpp src/vm_agent.h
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@.tmp src/vm_agent.cpp && mv $@.tmp $@

# The GPU update (task 00011): nvcc + cuBLAS, native code for the RTX 3090 (sm_86) and RTX 4090 (sm_89). Optional: not part of `all`; its tests skip
# without it.
NVCC = nvcc
libvmcuda.so: src/vm_cuda.cu src/vm_cuda.h src/vm_agent.h
	$(NVCC) -O3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC -shared -o $@.tmp src/vm_cuda.cu -lcublas && mv $@.tmp $@

test_vm_agent: src/test_vm_agent.cpp src/vm_agent.cpp src/vm_agent.h
	$(CXX) $(CXXFLAGS) -o $@ src/test_vm_agent.cpp

bench_vm_agent: src/bench_vm_agent.cpp src/vm_agent.cpp src/vm_agent.h
	$(CXX) $(CXXFLAGS) -o $@ src/bench_vm_agent.cpp

# C++ unit tests (GEMM, ops, causality, acting == training, checkpoints)
test: test_vm_agent
	./test_vm_agent

# Python tests: world, rewards, information leaks, JS equivalence, PyTorch reference equivalence
pytest: libvmagent.so
	$(PY) -m pytest tests -q

test_all: test pytest

bench: bench_vm_agent
	./bench_vm_agent

# Setup: venv with numpy, pillow, matplotlib, pytest and CPU torch (torch is only used by the
# reference-equivalence test, never by the agent)
setup: venv
venv: venv/bin/python
venv/bin/python:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# Random-policy baselines
baseline:
	$(PY) random_baseline.py

clean:
	rm -f libvmagent.so test_vm_agent bench_vm_agent

.PHONY: all test pytest test_all bench setup venv baseline clean
