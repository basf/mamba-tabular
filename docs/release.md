# Build and release

The document outlines the steps to build and release the `mambular` package. At this point, it is assumed that the development and testing of the package have been completed successfully.

## 1. Test documentation
It is expected from the contributor to update the documentation as an when required along side the change in source code. Please use the below process to test the documentation:

```sh
cd mambular/docs/

make doctest
```
Fix any docstring related issue, then proceed with next steps.

## 2. Version update
The package version is mantained in `mambular/__version__.py` file. Increment the version according to the changes such as patch, minor, major or all.

## 3. Release
We use git flow for the package release.

- If you don't have git flow installed, you can follow the instructions [here](https://skoch.github.io/Git-Workflow/) to install it.
- Initialize git flow in the repository using the below command, if not already done.
- Use default options for the git flow initialization.

```sh
git flow init
```
- Start a new release using the below command. Replace `<version>` with the version number you want to release.
- The version number should be the same as the one you updated in the `__version__.py` file.
- The version number should be in the format `major.minor.patch`. For example, `v0.0.1`.
- The version number should be prefixed with a `v` as shown in the example. Otherwise, the pipeline to publish the package to PyPi will fail.

```sh
git flow release start v0.0.1
```
- A new branch is created from the `develop` branch. This new branch is named according to the convention `release/<version>`. For example, if you run `git flow release start v0.0.1`, the new branch will be `release/v0.0.1`.
- The current working branch switches to the newly created release branch. This means any new commits will be added to the release branch, not the `develop` branch.
- This new branch is used to finalize the release. You can perform tasks such as version number bumps, documentation updates, and final testing.

Once you are satisfied with changes, use the below command to finish the release.
```sh
 git flow release finish v0.0.1
```

- It will Merges the release branch into `main` (or `master`).
- Tags the release with the version number.
- Merges the release branch back into `develop`.
- Deletes the release branch.

Finally, push the commits and tags to origin.

```sh
git push origin --tags
```

This will publish the tags to the remote repository.

Additionally, the latest documents are published as follows:
- A workflow is triggered from the `main` (or `master`)  to publish the documentation via readthedocs.
- Documentation is published to readthedocs and can be accesseed at [mambular.readthedocs.io](https://mambular.readthedocs.io/en/latest/).


## 4. Publish package to PyPi

The package is published to PyPi using GitHub Actions. The workflow is triggered when a new tag is pushed to the repository. The workflow will build the package, upload it to PyPi.

## 5. GitHub Release

Create a new release on GitHub with the version number and release notes. The release notes should include the changes made in the release.
