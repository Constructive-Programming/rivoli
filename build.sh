#!/bin/bash
set -euo pipefail
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "rolibri $VERSION — building with rocm feature"
cargo build --release --features rocm
echo "binary: target/release/rolibri"
