//! Mock model offline; optional live call when `OPENAI_API_KEY` is set.
//!
//! ```bash
//! cargo run -p agent-runtime --example hello_model
//!
//! OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... \
//!   cargo run -p agent-runtime --example hello_model
//! ```

use agent_runtime::{
    GenerateOptions, Message, MockModel, Model, ModelError, OpenAiCompatibleModel,
};

fn ask_generic<M: Model>(model: &M, prompt: &str) -> Result<String, ModelError> {
    Ok(model.generate_text(prompt)?.text_content())
}

fn ask_dyn(model: &dyn Model, prompt: &str) -> Result<String, ModelError> {
    Ok(model.generate_text(prompt)?.text_content())
}

fn main() -> Result<(), ModelError> {
    let mock = MockModel::always("mock", "ownership is a tree, not a graph");

    println!("model: {}", mock.name());
    println!("generic: {}", ask_generic(&mock, "what is ownership?")?);
    println!("dyn:     {}", ask_dyn(&mock, "what is ownership?")?);

    let scripted = MockModel::new("scripted")
        .reply_text("first")
        .reply_text("second");
    println!(
        "queued: {} then {}",
        scripted.generate_text("a")?.text_content(),
        scripted.generate_text("b")?.text_content()
    );

    match OpenAiCompatibleModel::from_env() {
        Ok(live) => {
            let messages = [Message::user("Reply with exactly: pong")];
            let options = GenerateOptions::default();
            match live.generate(&messages, &[], &options) {
                Ok(response) => println!("live {}: {}", live.name(), response.text_content()),
                Err(error) => {
                    println!("live call failed (expected if the key/URL is wrong): {error}")
                }
            }
        }
        Err(_) => {
            println!("no OPENAI_API_KEY; skipped live call");
        }
    }

    Ok(())
}
