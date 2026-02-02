<p align="center">
  <img src="docs/assets/logo-clean.png" width="100" alt="Mimir">
</p>

<h1 align="center">Mimir</h1>

<p align="center">
  <a href="https://github.com/stensmir/mimir/releases/latest"><img src="https://img.shields.io/github/v/release/stensmir/mimir?style=flat-square&color=4ade80" alt="Release"></a>
  <a href="https://github.com/stensmir/mimir/releases"><img src="https://img.shields.io/github/downloads/stensmir/mimir/total?style=flat-square&color=4ade80" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/macOS-Apple_Silicon-4ade80?style=flat-square&logo=apple&logoColor=white" alt="macOS">
</p>

Mimir is an offline voice-to-text app for macOS. It runs entirely on your machine — no cloud, no tracking, no subscriptions. Powered by NVIDIA Parakeet and OpenAI Whisper, both running locally via ONNX Runtime.

Use it as a global dictation tool, a file transcriber, or connect it to your AI workflow via the built-in MCP server.

<p align="center">
  <img src="docs/assets/screenshot-home.png" width="700" alt="Mimir">
</p>

## Get started

Download the latest DMG from [Releases](https://github.com/stensmir/mimir/releases/latest), open it, drag Mimir to Applications. Done.

> Windows & Linux — coming soon.

## MCP integration

Connect Mimir to Claude Desktop, Cursor, or any MCP-compatible client:

```json
{
  "mcpServers": {
    "mimir": {
      "command": "mimir",
      "args": ["mcp"]
    }
  }
}
```

## Open source

Mimir is currently in early access. We're stabilizing the codebase and fixing bugs. Once the core is solid, we'll open the source. Star the repo to get notified.

## Feedback

Found a bug or have a feature request? [Leave feedback](https://stensmir.github.io/mimir/feedback.html) — help us make Mimir better.

## Website

[stensmir.github.io/mimir](https://stensmir.github.io/mimir/)
