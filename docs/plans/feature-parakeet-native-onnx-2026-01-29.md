# Implementation Plan: Parakeet Native ONNX Inference

## Technical Context

- **Language**: Rust 1.75+ (stable toolchain)
- **Key Crate**: parakeet-rs 0.3 (ONNX Runtime bindings for Parakeet TDT)
- **Runtime**: ONNX Runtime (bundled via parakeet-rs)
- **Framework**: Tauri v2
- **Model**: nvidia/parakeet-tdt-0.6b-v3 (ONNX export, ~600MB)

## Architecture

### Before (Python Server)
```
Audio → HTTP POST → Python Server (NeMo) → HTTP Response → Rust
```

### After (Native ONNX)
```
Audio → parakeet-rs (ONNX Runtime) → Result directly in Rust
```

## Critical Files

| File | Change |
|------|--------|
| `Cargo.toml` / `src-tauri/Cargo.lock` | Add parakeet-rs dependency |
| `src/transcription/parakeet.rs` | Rewrite: HTTP client → direct ONNX inference |
| `src-tauri/src/parakeet_commands.rs` | Remove server lifecycle, add model download/delete |
| `src-tauri/src/main.rs` | Update Tauri command registration |
| `src/config/preferences.rs` | Update config types (remove server fields) |
| `src-tauri/src/dictation_manager.rs` | Update to use new parakeet API |
| `ui/src/modules/parakeet.js` | Remove Python/server UI, add ONNX model UI |
| `scripts/parakeet_server.py` | Delete |
| `src-tauri/tauri.conf.json` | Remove parakeet_server.py from resources |

## Phases

### Phase 1: Setup
- Add parakeet-rs to Cargo.toml with ONNX features

### Phase 2: Core Implementation
- Rewrite `parakeet.rs` for direct ONNX inference
- Rewrite `parakeet_commands.rs`: model download/delete instead of server start/stop

### Phase 3: Integration
- Update config types in preferences.rs
- Update main.rs command registration
- Update dictation_manager.rs

### Phase 4: UI
- Update parakeet.js to remove Python/server references

### Phase 5: Cleanup
- Delete `scripts/parakeet_server.py`
- Remove from `tauri.conf.json` resources

## Risks

| Risk | Mitigation |
|------|------------|
| parakeet-rs maturity (v0.3) | Pin version, test thoroughly |
| ONNX model size (~600MB) | Show download progress, support retry |
| macOS ONNX Runtime compatibility | Test on both Intel and Apple Silicon |

## Constitution Check

- [x] No new Python dependencies introduced
- [x] Model storage follows XDG conventions
- [x] Existing transcription trait interface preserved
- [x] UI changes are backward-compatible (install flow unchanged for user)
