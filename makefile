sources = src/ commands/ scripts/

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix