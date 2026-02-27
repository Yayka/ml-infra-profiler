.PHONY: build-linux build-darwin test

# Build linux/amd64 binary (for deployment on Linux compute nodes)
build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o bin/agent-linux ./cmd/agent

# Build native macOS binary (for local dev/testing on macOS)
build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o bin/agent-darwin ./cmd/agent

test:
	go test ./...
