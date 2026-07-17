#!/bin/bash
set -euo pipefail
cargo fmt --check
cargo clippy --all-targets --features rocm -- -D warnings
cargo test --features rocm
