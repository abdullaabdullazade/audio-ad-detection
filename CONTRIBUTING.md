# Contributing

Thank you for contributing to audio-ad-detection.

Before opening a pull request:

1. Run gofmt on changed Go files.
2. Run go test ./... and go vet ./....
3. Run race tests when changing concurrency or shared state.
4. Add focused tests for detector behaviour.
5. Document benchmark changes and the dataset/command used.

Do not commit private recordings, credentials, generated archives, or user-uploaded
reference clips. Security-sensitive issues should follow SECURITY.md.

