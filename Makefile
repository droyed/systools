.PHONY: help clean

help:
	@echo "Available targets:"
	@echo "  clean   Remove *.pyc files and __pycache__ directories"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
