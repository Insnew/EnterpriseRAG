# 镜像配方：api 与 ui 共用一个镜像（compose 里用不同 command 启动）
#
# 构建：docker compose build
# 依赖下载走 pyproject.toml 里配置的阿里云镜像（[[tool.uv.index]]），国内快

# 国内直连 Docker Hub 不稳定，基础镜像默认走加速站；国外网络可覆盖：
#   docker compose build --build-arg BASE_IMAGE=python:3.12-slim
ARG BASE_IMAGE=docker.1panel.live/library/python:3.12-slim
FROM ${BASE_IMAGE}

# uv 是 Rust 写的包管理器，pip 装一个二进制即可（走阿里云镜像加速）
RUN pip install --no-cache-dir uv -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

# 先只拷依赖清单 → 装依赖。代码改动时依赖层有缓存，重建秒级。
# README.md 必须带上：pyproject 声明 readme 字段，hatchling 打包时校验存在
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# 再拷代码（app/ui/eval/samples/scripts 按需；tests 不需要进镜像）
COPY app ./app
COPY ui ./ui
COPY eval ./eval
COPY samples ./samples
COPY scripts ./scripts

# 数据目录（docstore.db / context_cache.db）挂载点
RUN mkdir -p /app/data

EXPOSE 8000 8501
