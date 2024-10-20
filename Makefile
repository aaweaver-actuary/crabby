format:
	cargo fmt --all 
	cargo clippy --all-targets --all-features -- -D warnings

coverage:
	cargo llvm-cov clean
	cargo llvm-cov --lcov --output-path lcov.info
	cargo llvm-cov report --html

test:
	cargo test

tree:
	rm -f tree
	tree -I target > tree

build:
	cargo build --release 