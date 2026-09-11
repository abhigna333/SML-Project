precommit_update:
	pre-commit autoupdate

precommit_run:
	pre-commit install
	pre-commit run --all-files
