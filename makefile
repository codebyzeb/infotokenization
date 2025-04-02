sources = src/ commands/

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix