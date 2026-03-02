.PHONY: build-linux build-darwin test

# Build native macOS binary (for local dev/testing on macOS)
build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o bin/agent-darwin ./cmd/agent

test:
	go test ./...
