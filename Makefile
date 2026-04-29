.PHONY: setup run dev clean

setup:
	uv venv && . .venv/bin/activate && uv sync

run:
	uv run uvicorn main:app --reload

dev:
	uv run python main.py

clean:
	rm -rf __pycache__ .venv