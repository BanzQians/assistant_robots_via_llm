FROM registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0

SHELL ["/bin/bash", "-lc"]

RUN source /isaac-sim/.venv/bin/activate \
    && . /isaac-sim/python.env.init \
    && python -m pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        'numpy<1.27' \
        'pin<4'
