# The five benchmark suites on x86 Ubuntu with every external system:
# Ceres, g2o and GTSAM from Ubuntu's packages, SymForce v0.11.0 from
# source, factrs and tiny-solver through cargo. Everything is built
# while the image builds, so a run inside only runs.
#
# From the repository root:
#
#   docker build -f benchmarks/Dockerfile -t arael-bench .
#   docker run -it --rm arael-bench
#
# then inside, for example:
#
#   cd /arael/benchmarks/pgo  && ROUNDS=32 cargo run --release
#   cd /arael/benchmarks/slam && SLAM_POSES=300 ROUNDS=32 cargo run --release --bin slam-bench
#   cd /arael/benchmarks/bal  && BAL_ONLY=372 ROUNDS=4 cargo run --release
#
# The harness pins itself to one core and caps every thread pool it
# spawns at BENCH_THREADS (1 unless set), so a threaded run is
# BENCH_THREADS=8 in front of the command. GTSAM_PYTHON is preset for
# the pgo rows that run GTSAM through Python.
#
# Everything under /arael is image content and goes with the container.
# To keep a run's output, mount a host directory and write there:
#
#   docker run -it --rm -v ~/arael-runs:/runs arael-bench
#   cd /arael/benchmarks/pgo && ROUNDS=32 cargo run --release 2>&1 | tee /runs/pgo.$(date +%Y%m%d).txt
#
# The files come out owned by the container user's id, 1000 (the
# image's ubuntu user). On a host where that is not your id, add
# --user $(id -u):$(id -g) to docker run and start the built binaries
# directly (./target/release/pgo-bench and so on): cargo run would then
# fail on the target directory, which belongs to ubuntu. The runs run
# as that user, with passwordless sudo for installing more inside a
# container.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config git curl ca-certificates bzip2 \
        sudo vim \
        python3 python3-venv python3-numpy \
        libeigen3-dev libsuitesparse-dev \
        libceres-dev libg2o-dev \
        libgtsam-dev python3-gtsam libboost-all-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust: the toolchain the repository pins in rust-toolchain.toml.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain 1.94.0

# SymForce v0.11.0: the three C++ libraries the runners link, built from
# source into build/temp.* where the suites' cmake globs for them (its
# own install step ships neither symforce_opt nor symforce_cholesky).
# The venv serves only the bundled skymarshal codegen that the C++
# build runs; the Python package is not installed.
ENV SYMFORCE_DIR=/opt/symforce
RUN git clone --depth 1 --branch v0.11.0 --recursive \
        https://github.com/symforce-org/symforce $SYMFORCE_DIR \
    && python3 -m venv $SYMFORCE_DIR/.venv \
    && $SYMFORCE_DIR/.venv/bin/pip install --no-cache-dir -e $SYMFORCE_DIR/third_party/skymarshal \
    && cmake -S $SYMFORCE_DIR -B $SYMFORCE_DIR/build/temp.docker \
        -DCMAKE_BUILD_TYPE=Release \
        -DSYMFORCE_PYTHON_OVERRIDE=$SYMFORCE_DIR/.venv/bin/python \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$SYMFORCE_DIR/build/temp.docker \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$SYMFORCE_DIR/build/temp.docker \
        -DSYMFORCE_BUILD_TESTS=OFF -DSYMFORCE_BUILD_EXAMPLES=OFF \
        -DSYMFORCE_BUILD_CC_SYM=OFF -DSYMFORCE_BUILD_SYMENGINE=OFF \
    && cmake --build $SYMFORCE_DIR/build/temp.docker \
        --target symforce_opt symforce_gen symforce_cholesky -j"$(nproc)"

# From here on the image's ubuntu user (uid 1000): the repository is
# copied as its own, so the builds and later runs write their target
# directories and lock files in place, and the cargo and rustup homes
# are its to write. Passwordless sudo, for installing more inside a
# running container.
RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && chown -R ubuntu:ubuntu /opt/cargo /opt/rustup
USER ubuntu

# The repository (see .dockerignore at its root), then the datasets the
# suites read from files: the four pose graphs and the four Ladybug
# problems, with the cleaned 1723 the bal README describes.
COPY --chown=ubuntu:ubuntu . /arael
WORKDIR /arael/benchmarks
RUN ./pgo/fetch_datasets.sh && ./bal/fetch_datasets.sh \
    && cd bal/datasets \
    && for p in 138-19878 372-47423 1723-156502; do \
           [ -f problem-$p-pre.txt ] || { \
               curl -sL -o problem-$p-pre.txt.bz2 \
                   https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-$p-pre.txt.bz2 \
               && bunzip2 problem-$p-pre.txt.bz2; }; \
       done \
    && { [ -f problem-1723-156502-clean.txt ] \
         || python3 ../clean_bal.py problem-1723-156502-pre.txt --write problem-1723-156502-clean.txt; }

# The C++ runners of every suite, the factrs f32 subprocess crate the
# pgo rows run, then the Rust benchmarks themselves.
RUN for b in pgo bal slam loc plane; do \
        cmake -S $b/cpp -B $b/cpp/build -DCMAKE_BUILD_TYPE=Release -DSYMFORCE_DIR=$SYMFORCE_DIR \
        && cmake --build $b/cpp/build -j"$(nproc)" || exit 1; \
    done
RUN cd pgo/factrs32 && cargo build --release
RUN for b in pgo bal slam loc plane; do (cd $b && cargo build --release) || exit 1; done

ENV GTSAM_PYTHON=/usr/bin/python3
CMD ["/bin/bash"]
