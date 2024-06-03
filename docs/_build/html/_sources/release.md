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

```sh
git flow release start <version>
```
- A new branch is created from the `develop` branch. This new branch is named according to the convention `release/<version>`. For example, if you run `git flow release start 1.0.0`, the new branch will be `release/1.0.0`.
- The current working branch switches to the newly created release branch. This means any new commits will be added to the release branch, not the `develop` branch.
- This new branch is used to finalize the release. You can perform tasks such as version number bumps, documentation updates, and final testing.

Once you are satisfied with changes, use the below command to finish the release.
```sh
 git flow release finish <version>
```

- It will Merges the release branch into `main` (or `master`).
- Tags the release with the version number.
- Merges the release branch back into `develop`.
- Deletes the release branch.

Finally, push the commits and tags to origin.

```sh
git push origin --tags
```





