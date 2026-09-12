# agent-runtime-rs

A production-oriented agent runtime in Rust.

Architecture is inspired by [AgentScope Java 2.0](https://github.com/agentscope-ai/agentscope-java). This is not a line-by-line port — types and APIs are redesigned around Rust.

## Status

Early. What exists today:

- **Message** — `Role`, `Message`, `ContentBlock` (`Text` / `ToolCall` / `ToolResult`)
- **Model** — sync `Model` trait, `MockModel`, OpenAI-compatible Chat Completions client

Not yet: agent loop, tool runtime, middleware, session, sandbox.

## Layout

```text
crates/core/     library crate (modules grow here; not one crate per domain)
  src/message.rs
  src/model/
  examples/
```

## Build

```bash
cargo test -p agent-runtime
cargo run -p agent-runtime --example hello_message
cargo run -p agent-runtime --example hello_model
```

`hello_model` uses `MockModel` offline. A live call runs only when `OPENAI_API_KEY` is set. Compatible providers work via `OPENAI_BASE_URL` / `OPENAI_MODEL`.

## License

Apache-2.0
