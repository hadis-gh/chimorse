# Contributing to chimorse

Thank you for your interest in contributing to `chimorse`! We welcome contributions from computational physicists, chemists, software developers, and researchers of all experience levels.

---

## Table of Contents
* [Reporting Bugs](#reporting-bugs)
* [Suggesting Enhancements](#suggesting-enhancements)
* [Submitting Code and Pull Requests](#submitting-code-and-pull-requests)
* [Support and Questions](#support-and-questions)

---

## Reporting Bugs

If you find a bug, unexpected mathematical behavior, or an issue with the documentation, please let us know! 

1. **Search existing issues:** Check the GitHub Issues page to see if the problem has already been reported.
2. **Open a new issue:** If it is a new bug, open an issue on our GitHub repository. Please provide:
   * A clear, descriptive title.
   * Steps to reproduce the unexpected behavior.
   * A minimal working example (MWE) or a tiny snippet of the code/data causing the error.
   * Your operating system and Python version.

---

## Suggesting Enhancements

We are always looking to make `chimorse` more versatile! If you have ideas for new features—such as support for alternative potential-energy functional forms, different symmetry mappings, or additional visualization options:

1. Open a **Feature Request** on the GitHub Issues page.
2. Describe the feature you would like to see, why it is useful, and how you envision it integrating into the current workflow.
3. If applicable, link to any relevant scientific literature or equations.

---

## Submitting Code and Pull Requests

If you would like to contribute code or documentation changes directly, please use the following workflow:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone [https://github.com/hadis-gh/chimorse.git](https://github.com/hadis-gh/chimorse.git)
   cd chimorse
   ```
1. **Create a new branch** for your feature or bug fix:

    ```bash
    git checkout -b feature/my-new-feature
    ```
2. **Set up the test environment**:
Install the package locally in editable mode along with its testing dependencies:

    ```bash
    python -m pip install -e ".[test]"
    ```
3. **Write your code & tests**:

    - If you add new mathematical functions, fitting routines, or modules, please add corresponding tests in the `tests/` directory.

    - Ensure your code follows the existing style and architecture of the package.

4. **Run the tests locally** to verify everything passes:

    ```bash
    pytest
    ```
5. **Commit and push** your changes to your fork, and open a **Pull Request (PR)** against our `main` branch.

---

## Support and Questions

* If you have questions about using `chimorse` for your research, setting up your input datasets, or interpreting your fitted models, we are happy to help.

* For general usage questions or configuration issues, please feel free to open an **Issue** or start a **Discussion** on our GitHub repository.

* For personal inquiries or specific collaboration proposals, you can contact the primary author directly:
    **Hadis Ghodrati Saeini** (hadis.ghodrati-saeini@physik.tu-chemnitz.de)