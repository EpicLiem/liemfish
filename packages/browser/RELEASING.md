# Browser package releases

`.github/workflows/publish-browser.yml` publishes `@epicliem/liemfish-browser` when a `browser-vVERSION` tag is pushed. The tag must match package.json. A manual workflow run builds, tests, packs, and checks publication contents without publishing.

The workflow uses a GitHub-hosted runner and npm OIDC trusted publishing. Configure the npm trusted publisher for GitHub owner `EpicLiem`, repository `liemfish`, workflow filename `publish-browser.yml`, no environment, with direct `npm publish` allowed. The repository URL in package.json must match this repository.

For a new release:

```sh
cd packages/browser
npm version patch --no-git-tag-version
cd ../..
git add packages/browser/package.json packages/browser/package-lock.json
git commit -m "Release browser package VERSION"
git tag -a browser-vVERSION -m "Browser package VERSION"
git push origin main browser-vVERSION
```

Replace VERSION with the new package version. Never reuse an already published npm version. Engine tests use the selected ONNX pair, restored from Git LFS during the workflow.

To validate CI without publishing:

```sh
gh workflow run publish-browser.yml --ref main
```

Npm requires a new trusted-publisher configuration to complete its first successful publish within two days. If that window passes before the first tagged release, recreate the connection in npm package settings (or with `npm trust github @epicliem/liemfish-browser --file publish-browser.yml --repo EpicLiem/liemfish --allow-publish --yes`). A manual dry-run does not activate the connection. See https://docs.npmjs.com/trusted-publishers/.
