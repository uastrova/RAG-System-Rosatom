# Установка зависимостей на Ubuntu 

### 1. Установка зависимостей

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-pip
```

#### 1.1. CUDA-сборка __(рекомендуется)__

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

Это официальный способ собрать backend под NVIDIA GPU. В документации также указано, что можно отключить `GGML_NATIVE`, если нужна более универсальная сборка, и вручную задавать `CMAKE_CUDA_ARCHITECTURES` под compute capability GPU. ([GitHub][2])

Если несколько версий CUDA, можно явно указать `nvcc`, например:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-11.7/bin/nvcc \
  -DCMAKE_INSTALL_RPATH="/opt/cuda-11.7/lib64;\$ORIGIN" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build build --config Release -j 8
```

Такой сценарий прямо описан в build guide. ([GitHub][2])


#### 1.2. Можно собрать CPU-версию с ускорением через OpenBLAS:

```bash
sudo apt install -y libopenblas-dev
```

__CPU-сборка__

Официальная базовая команда такая:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8
```

Это соответствует официальному build guide. Для CPU также можно собирать с BLAS/OpenBLAS. ([GitHub][2])

__CPU + OpenBLAS__

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j 8
```

OpenBLAS в документации указан как CPU-ускорение через BLAS. ([GitHub][2])


## Как поднять API-сервер

`llama-server` — это встроенный HTTP server. В официальном README у него заявлены:

* OpenAI-compatible `chat completions`, `responses`, `embeddings`
* parallel decoding
* continuous batching
* web UI
* function calling / tool use
* multimodal support (где поддерживается)
* monitoring endpoints ([GitHub][3])

### Базовый запуск

```bash
./build/bin/llama-server \
  -m /models/my_model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -c 8192
```

---



# Пошаговая инструкция для Ubuntu + NVIDIA GPU

### Шаг 1. Собрать под CUDA

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-pip

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 8
```

([GitHub][2])

### Шаг 2. __Опиционально.__ Можно убедиться в правильности установки, проверить на HF GGUF

```bash
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -cnv -ngl 999
```

([GitHub][1])

### Шаг 3. Поднять сервер

```bash
./build/bin/llama-server \
  -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -c 8192
```

([GitHub][1])

### Шаг 4. Проверить API

```bash
curl http://127.0.0.1:8080/health
```
и потом уже `POST` на `/v1/chat/completions`.



[1]: https://github.com/ggml-org/llama.cpp "GitHub - ggml-org/llama.cpp: LLM inference in C/C++ · GitHub"
[2]: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md "llama.cpp/docs/build.md at master · ggml-org/llama.cpp · GitHub"
[3]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp/tools/server/README.md at master · ggml-org/llama.cpp · GitHub"


