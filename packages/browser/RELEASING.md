# Browser package releases

`.github/workflows/publish-browser.yml` publishes `@epicliem/liemfish-browser` when a `browser-vVERSION` tag is pushed. The tag must match package.json. A manual workflow run builds, tests, packs, and checks publication contents without publishing.

The workflow uses a GitHub-hosted runner and npm OIDC trusted publishing. Configure the npm trusted publisher for GitHub owner `EpicLiem`, repository `liemfish`, workflow filename `publish-browser.yml`, no environment, with direct `npm publish` allowed. No NPM_TOKEN secret is used. The repository URL in package.json must match this repository.

For a new release:

```sh
cd packages/browser
npm version patch --no-git-tag-version
cd ../..
git add packages/browser/package.json packages/browser/package-lock.json
git commit -m "Release browser package VERSION"
git tag browser-vVERSION
git push origin main browser-vVERSION
```

Replace VERSION with the new package version. Never reuse an already published npm version. Engine tests include the exact selected ONNX pair; the workflow restores those two LFS files, but never packages weights or training data. Runtime browser gameplay remains a separate consumer acceptance check.

To validate CI without publishing:

```sh
gh workflow run publish-browser.yml --ref main
```
