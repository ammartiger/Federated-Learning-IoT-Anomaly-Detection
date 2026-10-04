# ==============================================================================
# Dockerfile: Exact Paper Environment Replication
# Paper: "Federated-Learning-Based Anomaly Detection for IoT Security Attacks"
# Authors: Mothukuri et al. (IEEE IoT Journal, Feb 2022)
# Original Paper Environment:
#   - Server: Lambda GPU Server
#   - Operating System: Ubuntu 18.04 LTS (cited as Ubuntu 18.0.0 in Section IV)
#   - Frameworks: PyTorch, Skorch, Scikit-Learn, Scapy, CICFlowmeter
# ==============================================================================

FROM ubuntu:18.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, Python 3, build essentials, and libpcap for PCAP parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    python3-setuptools \
    build-essential \
    libpcap-dev \
    tcpdump \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# Set working directory
WORKDIR /workspace

# Copy dependency requirements
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch and ML dependencies matching the paper stack
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    skorch \
    matplotlib \
    seaborn \
    tqdm \
    scapy \
    tabulate \
    pypdf \
    cicflowmeter

# Copy project source code and dataset
COPY . /workspace/

# Default command to run the full paper replication experiment
CMD ["python", "experiment.py", "--rounds", "8", "--local_epochs", "2", "--max_samples_per_class", "1500"]
