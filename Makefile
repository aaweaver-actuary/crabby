# Purpose: Makefile for the project


format:
	cargo fmt --all 
	cargo clippy --all-targets --all-features -- -D warnings

coverage:
	cargo llvm-cov --lcov --output-path lcov.info

test:
	cargo test