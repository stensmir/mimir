# Windows Build Guide

Instructions for building Mimir on Windows.

## Prerequisites

### 1. Rust Toolchain

Download and install from [rustup.rs](https://rustup.rs):

```powershell
# After installation, verify:
rustc --version   # Should be 1.75+
cargo --version
```

### 2. Visual Studio Build Tools

Required for compiling native dependencies (whisper-rs, rusqlite).

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install with "Desktop development with C++" workload
3. Restart terminal after installation

### 3. Node.js

Required for building the UI:

1. Download LTS from [nodejs.org](https://nodejs.org)
2. Install and verify:

```powershell
node --version   # Should be 18+
npm --version
```

### 4. (Optional) CUDA Toolkit

For GPU-accelerated Whisper transcription:

1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install and add to PATH

## Building

### Clone and Setup

```powershell
git clone https://github.com/YOUR_USERNAME/mimir.git
cd mimir
git checkout 015-windows-port

# Install UI dependencies
cd ui
npm install
cd ..
```

### Development Build

For testing with hot-reload:

```powershell
cargo tauri dev
```

### Release Build

Creates distributable installer:

```powershell
cargo tauri build
```

Output location: `src-tauri/target/release/bundle/`

- `nsis/` - NSIS installer (.exe)
- `msi/` - MSI installer (if configured)

## Troubleshooting

### "LINK : fatal error LNK1181: cannot open input file"

Visual Studio Build Tools not installed correctly. Reinstall with C++ workload.

### whisper-rs compilation errors

1. Ensure Visual Studio Build Tools are installed
2. For CUDA support, ensure CUDA Toolkit is in PATH
3. Try without CUDA first (CPU mode works out of box)

### "npm not found"

Add Node.js to PATH or restart terminal after installation.

### SQLite compilation errors

Visual Studio Build Tools required. The `rusqlite` crate compiles SQLite from source.

## Testing

After successful build, test the following:

1. **Voice Dictation**: Press hotkey (Ctrl+Cmd+D), speak, verify text appears
2. **Auto-Launch**: Enable in Settings, restart Windows, verify Mimir starts
3. **Parakeet ASR**: If Python 3.10+ installed, test Parakeet engine installation
4. **Hotkey Config**: Change hotkey in Settings, verify new hotkey works

## Platform-Specific Notes

### Text Injection

Windows uses clipboard + Ctrl+V for text injection (same approach as macOS with Cmd+V).

### Auto-Launch

Uses Windows Registry (`HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run`).

### Permissions

Unlike macOS, Windows doesn't require explicit accessibility or microphone permissions for desktop apps. The app assumes permissions are granted.

## Architecture

Key platform abstractions:

| Component | macOS | Windows |
|-----------|-------|---------|
| Hotkey Manager | CGEventHotkeyManager | RdevHotkeyManager |
| Text Injector | MacOSInjector (Cmd+V) | WindowsInjector (Ctrl+V) |
| Auto-Launch | SMAppService | Registry |
| Permissions | AVFoundation, AXIsProcessTrusted | Granted by default |
