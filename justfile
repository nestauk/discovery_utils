install:
    poetry install --with lint,test
    poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
