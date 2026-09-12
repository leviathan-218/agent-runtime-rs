//! Smallest useful demo of the message model.
//!
//! ```bash
//! cargo run -p agent-runtime --example hello_message
//! ```

use agent_runtime::{Message, Role};

fn summarize(message: &Message) -> String {
    format!("{} chars from {}", message.text().len(), message.role)
}

fn consume(message: Message) {
    println!("consumed: {message}");
}

fn main() {
    let user = Message::user("What is ownership in Rust?");
    let system = Message::system("You are a concise assistant.");
    let assistant = Message::assistant("Ownership means each value has one owner.");

    println!("roles: {}, {}, {}", user.role, system.role, assistant.role);
    assert_eq!(user.role, Role::User);

    println!("summary: {}", summarize(&user));
    println!("still here: {user}");

    consume(user);
}
