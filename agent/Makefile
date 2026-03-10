.PHONY: build-darwin build-linux build-linux-ib build-linux-nvidia build-linux-full test vet-linux-ib vet-linux-nvidia

# Build native macOS binary (for local dev/testing on macOS)
build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o bin/agent-darwin ./cmd/agent

# Linux: Ethernet only (CGO_ENABLED=0 — no DCGM)
build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o bin/agent-linux ./cmd/agent

# Linux: Ethernet + InfiniBand sysfs collector
build-linux-ib:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -tags infiniband -o bin/agent-linux-ib ./cmd/agent

# Linux: Ethernet + NVLink/DCGM (requires CGO_ENABLED=1 and DCGM headers)
build-linux-nvidia:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags nvlink -o bin/agent-linux-nvidia ./cmd/agent

# Linux: Ethernet + InfiniBand + NVLink (full)
build-linux-full:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags "infiniband nvlink" -o bin/agent-linux-full ./cmd/agent

# Run unit tests (darwin tags active; linux-tagged files are skipped)
test:
	go test ./...

# Compile-check infiniband-tagged files without running (needs GOOS=linux)
vet-linux-ib:
	GOOS=linux go vet -tags infiniband ./...

# Compile-check nvlink-tagged files without running (needs GOOS=linux + CGO)
vet-linux-nvidia:
	CGO_ENABLED=1 GOOS=linux go vet -tags nvlink ./...
