# Contributing to the "Hybrid Deep Representation Learning Framework for Software Fault Prediction from Syntactic and Semantic Code Graphs"

In the context of a traineeship project in collaboration between the **Université Côte d'Azur (UCA)** and the **Vietnam Korea University of Information and Communication Technology (VKU - Da Nang, Vietnam)**.

First of all, thank you for taking the time to contribute!
We are excited to welcome contributions to enhance our project. This document provides guidelines and best practices for contributing to our repository, ensuring a smooth, clean, and collaborative development process.


### Table of Contents

* [How to Contribute](#how-to-contribute)
    * [Ask questions](#ask-questions)
    * [Branching strategy](#branching-strategy)
    * [Release strategy](#release-strategy)
    * [Create an Issue](#create-an-issue)
    * [Create a Milestone](#create-a-milestone)
    * [Issue Lifecycle](#issue-lifecycle)
    * [Milestone Lifecycle](#milestone-lifecycle)


### Code of Conduct

By participating you are expected to uphold the code of conduct that will be presented in this document.

In general (Python Standards - PEP 8):
- PascalCase to name classes (e.g., CFGBuilder).
- snake_case to name variables, functions, and files (e.g., parse_to_AST).
- test_*.py for unit test files.

Ensure your code is clean, modular, and well-commented, especially for complex graph parsing (AST, CFG, DDG) and deep learning architecture (DH-CNN).

More specific contribution rules are detailed in the sections below.

--------


### How to Contribute

#### Ask questions

If you have any questions, feel free to ask one of the contributors.

#### Branching strategy

We use the **Gitflow** branching strategy.

Our project comport two permanent branches:
* **main**: stores the official release history. Only stable code can be merged into this branch.
* **dev**: used as the integration branch where new features are merged.

For each new feature, create a dedicated branch from `dev`. Once the feature is implemented, open a Pull Request (PR) to merge it into `dev`.

Feature branch naming convention:

    feature-<number of the issue>-<name-of-the-issue>

If a bug is found in the `dev` branch following a feature implementation, create a fix branch and open a PR to merge it into `dev`.

Fix branch naming convention:

    fix-<Keyword-of-the-bug>

When several meaningful features have been implemented and the `dev` branch is stable, a new release can be created by merging `dev` into `main`.

/!\ Direct commits or merges into the `main` and `dev` branches are not allowed /!\
All changes must be introduced through a Pull Request and reviewed before being merged.

##### Commits

Each **Commit** must follow the format of _Conventional Commits_:

    <type>([optional scope]): <description>

    [optional body]

    [optional footer(s)]

Additionally, every commit can be referenced to a related issue by adding the following footer:

    #<issue-number>

Allowed commit types:
*    **feat** – a new feature is introduced with the changes
*    **fix** – a bug fix has occurred
*    **chore** – changes that do not relate to a fix or feature and don't modify src or test files (for example updating dependencies)
*    **refactor** – refactored code that neither fixes a bug nor adds a feature
*    **docs** – updates to documentation such as the README or other markdown files
*    **style** – changes that do not affect the meaning of the code, likely related to code formatting such as white-space, missing semicolons, and so on.
*    **test** – including new or correcting previous tests
*    **perf** – performance improvements
*    **ci** – continuous integration related
*    **build** – changes that affect the build system or external dependencies 
*    **revert** – reverts a previous commit

These are the rules that must be followed in the repository for more details please check [Conventional Commits][conventional-commits_link].

[conventional-commits_link]: https://www.conventionalcommits.org/en/v1.0.0/

##### Pull Requests

When creating a **Pull Request**, the following format must be respected:

**Title:**

    <type>([optional scope]): <purpose of the PR> #IssueNumber

**Content:**

    ### Description

    < Description of what this PR does.>

    ### Related Issue:

        closes #<number related issue>

    ### Additional Notes

    < Any additional information or context relevant to the PR.>

The types and scopes used in a PR must be the same as those used in commits.

When merging a PR, use the PR title as the merge commit title and the GitHub-generated description as the merge commit message.

#### Release Strategy

A new **Release** is created every time the `dev` branch is merged into `main`.

A release:
* Corresponds to a stable and validated state of the project
* Includes all completed and reviewed issues from the associated milestone
* Generates a new Git tag

#### Create an Issue

When creating an **issue**, respect the following format:

**Title**

    <short description of the issue>

**Description**

Provide a list of tasks:

````
# Tasks

    - Break down the work into actionable tasks
````

Then assign the issue to a milestone. Use an existing milestone that can contain the issue or create a new one following the format bellow.

#### Create a Milestone

To create a **Milestone** you will have to respect this format:

    Sprint<sprint-number>: <Description of the milestone>

And set a due date for the milestone.

#### Issue Lifecycle

* When an issue is first created, it is flagged `Backlog` waiting to be assigned to someone.
* When the issue is attributed to someone, the issue is flagged `Ready`.
* When work starts, it is flagged as `In Progress`.
* When completed and part of a PR, it is flagged as `In review`.
* Once the issue has been reviewed, it can be closed and flagged `Done`.

#### Milestone Lifecycle

When all issues in a milestone are marked `In review`, each issue is reviewed. Once validation is complete, all related pull requests are closed and a release is created by merging the `dev` branch into the `main` branch via a pull request.

### Contributors

| [**Sabra ESSALAH**](https://github.com/sabraess) | [**Matthieu TERRIER**](https://github.com/MatthieuTrr) |
| :---: | :---: |
| [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com) | [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com) |