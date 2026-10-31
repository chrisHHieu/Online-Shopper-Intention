.PHONY: install lint test fetch validate features train evaluate drift serve

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	python -m pytest -q

fetch:
	python -m shopper_intent fetch

validate:
	python -m shopper_intent validate

features:
	python -m shopper_intent features

train:
	python -m shopper_intent train

evaluate:
	python -m shopper_intent evaluate

drift:
	python -m shopper_intent drift

serve:
	python -m shopper_intent serve
