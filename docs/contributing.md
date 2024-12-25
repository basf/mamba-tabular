
## Contribution Guidelines

Thank you for considering contributing to our Python package! We appreciate your time and effort in helping us improve our project. Please take a moment to review the following guidelines to ensure a smooth and efficient contribution process.

### Code of Conduct

We kindly request all contributors to adhere to our Code of Conduct when participating in this project. It outlines our expectations for respectful and inclusive behavior within the community.

### Setting Up Development Environment

Before you start contributing to the project, you need to set up your development environment. This will allow you to make changes to the codebase, run tests, and build the documentation locally. The project uses `poetry` for dependency management and packaging. Along with that, `ruff` is used for source code formatting and linting.

To set up the development environment for this Python package, follow these steps:

1. Clone the repository to your local machine using the command:

```
git clone https://github.com/basf/mamba-tabular

cd mamba-tabular
```

2. Install tools required for setting up development environment:

- Install `poetry` for dependency management and packaging. You can install it using the following command or refer to the [official documentation](https://python-poetry.org/docs/) for more information.

```
pip install poetry
```

- Install `just` command runner. You can install it using the following command or refer to the [official documentation](https://just.systems/man/en/) for more information.

`justfile` in the source directory is used to define and run common tasks like testing, building, and formatting the codebase.

3. In case you are able to successfully install `poetry` and `just`, you can run the following command to install the dependencies and set up the development environment:

```
# it will install the dependencies as defined in the pyproject.toml file
# it will also install the pre-commit hooks

just install
```

In case you are not able to install `just`, you can follow the below steps to set up the development environment:

```
cd mamaba-tabular

poetry install

poetry run pre-commit install
```

If you need to update the documentation, please install the dependencies requried for documentation:

```
pip install -r docs/requirements_docs.txt
```

**Note:** You can also set up a virtual environment to isolate your development environment.


### How to Contribute

1. Create a new branch from the `develop` branch for your contributions. Please use descriptive and concise branch names.
2. Make your desired changes or additions to the codebase.
3. Ensure that your code adheres to [PEP8](https://peps.python.org/pep-0008/) coding style guidelines.
4. Write appropriate tests for your changes, ensuring that they pass.
    - `make test`
5. Update the documentation and examples, if necessary.
6. Build the html documentation and verify if it works as expected. We have used Sphinx for documentation, you could build the documents as follows:
    - `cd src/docs`
    - `make clean`
    - `make html`
7. Verify the html documents created under `docs/_build/html` directory. `index.html` file is the main file which contains link to all other files and doctree.

8. Commit your changes with a clear and concise commit message.
9. Submit a pull request from your branch to the development branch of the original repository.
10. Wait for the maintainers to review your pull request. Address any feedback or comments if required.
11. Once approved, your changes will be merged into the main codebase.

### Submitting Contributions

When submitting your contributions, please ensure the following:

- Include a clear and concise description of the changes made in your pull request.
- Reference any relevant issues or feature requests in the pull request description.
- Make sure your code follows the project's coding style and conventions.
- Include appropriate tests that cover your changes, ensuring they pass successfully.
- Update the documentation if necessary to reflect the changes made.
- Ensure that your pull request has a single, logical focus.

### Issue Tracker

If you encounter any bugs, have feature requests, or need assistance, please visit our [Issue Tracker](https://github.com/basf/mamba-tabular/issues). Make sure to search for existing issues before creating a new one.

### License

By contributing to this project, you agree that your contributions will be licensed under the LICENSE of the project.
Please note that the above guidelines are subject to change, and the project maintainers hold the right to reject or request modifications to any contributions. Thank you for your understanding and support in making this project better!
