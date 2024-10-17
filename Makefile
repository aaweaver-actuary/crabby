# Purpose: Makefile for the project
format:
	cargo fmt --all 
	cargo clippy --all-targets --all-features -- -D warnings

coverage:
	cargo llvm-cov clean
	cargo llvm-cov --lcov --output-path lcov.info
	cargo llvm-cov report --html

test:
	cargo test

build:
	cargo build --release 