# Multi-stage ROCm (gfx1151) image for rivoli + all offline artifact converters.
#
# Built on the minimal official ROCm 7.2.4 base (`rocm/dev-ubuntu-24.04:7.2.4`, ~1.2 GB),
# which ships libamdhip64 and ROCm device bitcode (ocml.bc + oclc_isa_version_1151.bc).
# hipcc is not a build input: every GPU kernel is Rust compiled on amdgcn-amd-amdhsa by
# the nightly toolchain pinned in `crates/kernels/rust-toolchain.toml`.

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2.4

FROM ${ROCM_IMAGE} AS builder

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG RIVOLI_OFFLOAD_ARCH=gfx1151
ENV RIVOLI_OFFLOAD_ARCH=${RIVOLI_OFFLOAD_ARCH} \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_TERM_COLOR=always

WORKDIR /src

# Install rustup (stable host compiler) plus the exact nightly channel + rust-src
# pinned by `crates/kernels/rust-toolchain.toml` for the amdgcn device crate.
COPY crates/kernels/rust-toolchain.toml crates/kernels/rust-toolchain.toml
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path; \
    KERNEL_CHANNEL="$(sed -n 's/^channel *= *"\([^"]*\)"/\1/p' crates/kernels/rust-toolchain.toml)"; \
    rustup toolchain install "${KERNEL_CHANNEL}" --profile minimal --component rust-src; \
    rustc --version; \
    rustup run "${KERNEL_CHANNEL}" rustc --version

COPY Cargo.toml Cargo.lock .jscpd.json ./
COPY crates ./crates
COPY docs ./docs

RUN cargo build --release --locked -p rivoli --features rocm --bins; \
    mkdir -p /out/bin; \
    for b in rivoli convert convert_v4 convert_glimmer convert_glimmer_drafter \
             convert_k3 convert_qwen convert_inkling convert_inkling_small \
             fp8_to_i4 add_indexer ppl; do \
      install -m755 "target/release/${b}" "/out/bin/${b}"; \
    done

FROM ${ROCM_IMAGE} AS runtime

LABEL org.opencontainers.image.title="rivoli" \
      org.opencontainers.image.description="Decode LLMs bigger than memory on AMD Strix Halo (gfx1151) — NVMe streaming overlapped with ROCm compute" \
      org.opencontainers.image.url="https://rivoli.constructive.dev" \
      org.opencontainers.image.source="https://github.com/Constructive-Programming/rivoli" \
      org.opencontainers.image.licenses="MIT"

ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=builder /out/bin/ /usr/local/bin/

ENTRYPOINT ["rivoli"]
CMD ["--help"]
