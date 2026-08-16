//! The output side: the merge that removes the `tool` role, and the render that turns turns
//! into the string the model is prefilled with.
//!
//! Split out of the reference's single 2822-line module (`old:src/artifact/dsv4_encoding.rs`)
//! by cohesion, under the 800-line cap. It depends on [`super::message`] and nothing depends on
//! it but [`super`]'s re-export — the reference draws the same seam with a banner comment.

// Ordered external-first and with the two `super::` lists merged, where `message.rs` puts
// `super::` first and lists `anyhow` before `serde_json`. Not style: jscpd normalizes
// identifiers, so the two files' import runs were a 29-token clone until one of them changed
// shape. Recorded because the obvious tidy-up — making every module's preamble identical —
// re-creates it.
use anyhow::{Result, bail, ensure};
use serde_json::Value;

use crate::tokenizer::{json_truthy, python_json};

use super::message::{AssistantTurn, Body, Instructions, Message, Tool, ToolCall};
use super::{
    ASSISTANT, DSML, EOS, LATEST_REMINDER, ReasoningEffort, THINK_CLOSE, THINK_OPEN, Task,
    ThinkingMode, USER,
};

// ============================================================
// Encoding options
// ============================================================

/// The knobs `encode_messages` takes, one per keyword argument of `encode_messages` in the
/// reference (minus `context` — see [`super`]'s header).
///
/// `Copy` with public fields and no builder: `EncodeOpts { reasoning_effort:
/// ReasoningEffort::High, ..EncodeOpts::new(ThinkingMode::Thinking) }` is the whole of what
/// three `#[must_use]` setters would buy.
#[derive(Debug, Clone, Copy)]
pub struct EncodeOpts {
    pub thinking: ThinkingMode,
    /// Strip reasoning from assistant turns *before* the last user message. Forced off — by
    /// the reference, not by us — as soon as any message declares a tool.
    pub drop_thinking: bool,
    /// The reference's `add_default_bos_token`.
    pub add_bos: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl EncodeOpts {
    /// The reference's defaults for everything except `thinking_mode`, which has no default
    /// there (it is a required positional) and gets none here: picking one silently is how a
    /// prompt ends up off-template.
    #[must_use]
    pub const fn new(thinking: ThinkingMode) -> Self {
        Self {
            thinking,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: ReasoningEffort::Low,
        }
    }
}

// ============================================================
// Preprocessing: tool messages become <tool_result> blocks in a user turn
// ============================================================

/// A user turn's content after merging: text the user typed, and results of tool calls.
#[derive(Debug, Clone)]
enum Block {
    Text(String),
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// A message that has been through [`merge_tool_messages`] — **there is no `Tool` variant**.
///
/// The reference guards this at runtime:
///
/// ```text
/// raise NotImplementedError("deepseek_v4 merges tool messages into user; please
///                            preprocess with merge_tool_messages()")
/// ```
///
/// Here the preprocessor's return type says it instead, so the renderer's match is total
/// with no unreachable arm to keep in sync.
#[derive(Debug, Clone)]
enum TurnBody {
    System(Instructions),
    Developer(Instructions),
    User(Vec<Block>),
    LatestReminder(String),
    Assistant(AssistantTurn),
}

#[derive(Debug, Clone)]
struct Turn {
    body: TurnBody,
    task: Option<Task>,
}

impl Turn {
    /// Whether this turn declares tools — the test that disables `drop_thinking`. An empty
    /// list is falsy in Python and must not count.
    fn declares_tools(&self) -> bool {
        match &self.body {
            TurnBody::System(i) | TurnBody::Developer(i) => !i.tools.is_empty(),
            TurnBody::User(_) | TurnBody::LatestReminder(_) | TurnBody::Assistant(_) => false,
        }
    }

    /// The renderer needs "is this a user-ish turn" in two places: the last-user scan and
    /// the generation-prompt decision. Developer counts as user in both — it is framed with
    /// `<｜User｜>`.
    fn is_user_like(&self) -> bool {
        match self.body {
            TurnBody::User(_) | TurnBody::Developer(_) => true,
            TurnBody::System(_) | TurnBody::LatestReminder(_) | TurnBody::Assistant(_) => false,
        }
    }
}

/// V4 has no standalone `tool` role: results are `<tool_result>` blocks inside a user turn.
///
/// Takes the messages by value. The reference deep-copies every message because it mutates
/// them in place; moving is the same guarantee for free.
fn merge_tool_messages(messages: Vec<Message>) -> Vec<Turn> {
    let mut merged: Vec<Turn> = Vec::with_capacity(messages.len());
    for Message { body, task } in messages {
        match body {
            Body::Tool {
                tool_call_id,
                content,
            } => absorb_tool_result(&mut merged, tool_call_id, content),
            Body::User { content } => absorb_user_text(&mut merged, content, task),
            Body::System(i) => merged.push(Turn {
                body: TurnBody::System(i),
                task,
            }),
            Body::Developer(i) => merged.push(Turn {
                body: TurnBody::Developer(i),
                task,
            }),
            Body::LatestReminder { content } => merged.push(Turn {
                body: TurnBody::LatestReminder(content),
                task,
            }),
            Body::Assistant(a) => merged.push(Turn {
                body: TurnBody::Assistant(a),
                task,
            }),
        }
    }
    merged
}

/// Push `block` into the last turn if it is a user turn, else open one.
///
/// The two merge rules differ by exactly one condition — a tool result joins a user turn even
/// when that turn already carries a task, and text does not — so they are two functions with one
/// shared tail rather than one function with a flag. See [`absorb_user_text`].
fn open_or_extend_user(merged: &mut Vec<Turn>, block: Block, task: Option<Task>, joinable: bool) {
    match merged.last_mut().map(|t| &mut t.body) {
        Some(TurnBody::User(blocks)) if joinable => blocks.push(block),
        // Spelled out rather than `_`: a new `TurnBody` variant must come back here and answer
        // whether a user block may merge into it. If a sixth variant ever makes this a jscpd
        // clone of something, factor a helper — do not reach for `_`.
        Some(
            TurnBody::User(_)
            | TurnBody::System(_)
            | TurnBody::Developer(_)
            | TurnBody::LatestReminder(_)
            | TurnBody::Assistant(_),
        )
        | None => merged.push(Turn {
            body: TurnBody::User(vec![block]),
            task,
        }),
    }
}

/// A `tool` message becomes a `<tool_result>` block on the preceding user turn — **whatever
/// task that turn carries**. The reference checks the task only on the text-merging branch, and
/// `super::tests::encode`'s `after_task` case pins the asymmetry.
fn absorb_tool_result(merged: &mut Vec<Turn>, tool_call_id: String, content: String) {
    let block = Block::ToolResult {
        tool_use_id: tool_call_id,
        content,
    };
    // `task: None` on the OPENED turn: a tool result never carries one.
    open_or_extend_user(merged, block, None, true);
}

/// A `user` message joins the preceding user turn only when that turn has NO task — otherwise
/// the task token would land mid-content.
fn absorb_user_text(merged: &mut Vec<Turn>, content: String, task: Option<Task>) {
    let joinable = matches!(merged.last(), Some(t) if t.task.is_none());
    open_or_extend_user(merged, Block::Text(content), task, joinable);
}

/// Reorder ONE user turn's `<tool_result>` blocks against `order`, in place.
///
/// Split out of [`sort_tool_results_by_call_order`] so the loop below is a dispatch and this
/// is the arithmetic — the reference has both in one body, which is the shape the code-health
/// gate refuses.
fn sort_one_turn(blocks: &mut [Block], order: &[(String, usize)]) {
    // Scanned in REVERSE: the reference assigns into a dict, so a repeated id
    // keeps the LAST index. Only reachable if a model emits two calls with the
    // same id, but a silent disagreement there is exactly the kind that never
    // gets found.
    let key = |b: &Block| match b {
        Block::ToolResult { tool_use_id, .. } => order
            .iter()
            .rev()
            .find(|(id, _)| id == tool_use_id)
            .map_or(0, |&(_, i)| i),
        Block::Text(_) => 0,
    };
    // Sort the tool results among themselves and drop them back into the same
    // slots: text blocks do not move. The reference does exactly this, which is
    // why it can interleave `[text, result, text, result]` without reordering
    // the prose.
    // COUNT before cloning. A single-result turn is the common case and it has
    // nothing to sort, but the clone would still deep-copy a whole search result
    // just to drop it — the same waste the user arm of `render_turn` calls out.
    let n_results = blocks
        .iter()
        .filter(|b| matches!(b, Block::ToolResult { .. }))
        .count();
    if n_results <= 1 {
        return;
    }
    let mut sorted: Vec<Block> = blocks
        .iter()
        .filter(|b| matches!(b, Block::ToolResult { .. }))
        .cloned()
        .collect();
    sorted.sort_by_key(key);
    let mut next = sorted.into_iter();
    for slot in blocks.iter_mut() {
        if matches!(slot, Block::ToolResult { .. })
            && let Some(b) = next.next()
        {
            *slot = b;
        }
    }
}

/// Reorder a user turn's `<tool_result>` blocks to match the order the preceding assistant
/// asked for them, so the model reads results in the order it made the calls.
///
/// Only when there is more than one result and the calls carried ids: a single result has
/// nothing to sort against, and an id-less call list would collapse every result to key 0.
/// A result whose id matches no call also sorts to 0 — with a stable sort, that leaves such
/// results in their original relative order at the front.
fn sort_tool_results_by_call_order(mut turns: Vec<Turn>) -> Vec<Turn> {
    // `(id, index)` pairs and NOT positions in a filtered list. The reference keys its dict
    // on the `enumerate` index over ALL of `tool_calls`, so an id-less call still consumes an
    // index and every call after it keeps its original number. Filtering first and then
    // taking a position renumbers them: with calls `[<no id>, "c1"]` and results
    // `[c1, unmatched]`, the reference emits unmatched-then-c1 (c1 is key 1, unmatched is the
    // default 0) and a renumbered list emits c1-then-unmatched (both key 0, stable order).
    // Caught in review 2026-08-05 by running the reference;
    // `super::tests::encode::tool_results_merge_into_the_user_turn` now pins it.
    let mut order: Vec<(String, usize)> = Vec::new();

    for turn in &mut turns {
        match &mut turn.body {
            TurnBody::Assistant(a) if !a.tool_calls.is_empty() => {
                order = a
                    .tool_calls
                    .iter()
                    .enumerate()
                    .filter(|(_, tc)| !tc.id.is_empty())
                    .map(|(i, tc)| (tc.id.clone(), i))
                    .collect();
            }
            TurnBody::User(blocks) if !order.is_empty() => sort_one_turn(blocks, &order),
            TurnBody::Assistant(_)
            | TurnBody::User(_)
            | TurnBody::System(_)
            | TurnBody::Developer(_)
            | TurnBody::LatestReminder(_) => {}
        }
    }
    turns
}

/// Index of the last `user` **or `developer`** turn, or `None` when there is none.
///
/// `None` is the reference's `-1`, and every comparison against it is true there — which is
/// why the callers below use `is_none_or` and not `map_or(false, …)`. It is not a
/// degenerate case that cannot happen: a conversation of `[system, assistant]` hits it, and
/// the reference then renders the assistant's reasoning and emits no `<｜Assistant｜>` at all.
fn last_user_index(turns: &[Turn]) -> Option<usize> {
    turns.iter().rposition(Turn::is_user_like)
}

/// Strip reasoning from assistant turns before the last user message, and drop developer
/// turns from before it entirely.
///
/// Only reached in thinking mode with `drop_thinking` still on — i.e. no tools anywhere.
fn drop_thinking_turns(turns: Vec<Turn>) -> Vec<Turn> {
    let last_user = last_user_index(&turns);
    turns
        .into_iter()
        .enumerate()
        .filter_map(|(idx, mut turn)| {
            let at_or_after_last_user = last_user.is_none_or(|l| idx >= l);
            match &mut turn.body {
                TurnBody::User(_) | TurnBody::System(_) | TurnBody::LatestReminder(_) => Some(turn),
                TurnBody::Assistant(a) => {
                    if !at_or_after_last_user {
                        a.reasoning_content.clear();
                    }
                    Some(turn)
                }
                // NOT in the reference's keep-list, so a developer turn before the last user
                // vanishes — instructions and tools with it.
                TurnBody::Developer(_) => at_or_after_last_user.then_some(turn),
            }
        })
        .collect()
}

// ============================================================
// Rendering
// ============================================================

/// The `## Tools` block, byte-for-byte from `TOOLS_TEMPLATE`.
///
/// Built with `format!` rather than pasted pre-substituted so that the `｜DSML｜` and
/// `<think>` constants in [`super`] are the single source of truth: change one and this block
/// follows, instead of drifting into a literal nobody re-reads.
fn render_tools(tools: &[Tool]) -> String {
    let schemas = tools
        .iter()
        .map(|t| python_json(&t.0))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. \
         You can invoke tools by writing a \"<{DSML}tool_calls>\" block like the following:\
         \n\n<{DSML}tool_calls>\n<{DSML}invoke name=\"$TOOL_NAME\">\n<{DSML}parameter \
         name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSML}parameter>\
         \n...\n</{DSML}invoke>\n<{DSML}invoke name=\"$TOOL_NAME2\">\n...\n</{DSML}invoke>\
         \n</{DSML}tool_calls>\n\nString parameters should be specified as is and set \
         `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass \
         the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled \
         (triggered by {THINK_OPEN}), you MUST output your complete reasoning inside \
         {THINK_OPEN}...{THINK_CLOSE} BEFORE any tool calls or final response.\n\nOtherwise, \
         output directly after {THINK_CLOSE} with tool calls or final response.\n\n### \
         Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined \
         tool name and parameter schemas to invoke tool calls.\n"
    )
}

/// What a `system` or `developer` turn appends after its instructions. Shared by both so
/// the two orderings cannot drift — tools first, response format second.
fn push_instructions_suffix(out: &mut String, i: &Instructions) {
    if !i.tools.is_empty() {
        out.push_str("\n\n");
        out.push_str(&render_tools(&i.tools));
    }
    if let Some(rf) = i.response_format.as_ref().filter(|v| json_truthy(v)) {
        out.push_str(
            "\n\n## Response Format:\n\nYou MUST strictly adhere to the following \
                      schema to reply:\n",
        );
        out.push_str(&python_json(rf));
    }
}

/// One tool call as DSML, appended in place: `\n`-joined parameters inside
/// `<｜DSML｜invoke>…</｜DSML｜invoke>`.
fn push_tool_call(out: &mut String, tc: &ToolCall) -> Result<()> {
    out.push_str(&format!("<{DSML}invoke name=\"{}\">\n", tc.name));
    push_dsml_parameters(out, tc)?;
    out.push_str(&format!("\n</{DSML}invoke>"));
    Ok(())
}

/// One `<｜DSML｜parameter>`.
///
/// A string value is written **raw** with `string="true"`; everything else as JSON with
/// `string="false"`. That asymmetry is the whole contract, and it is what lets
/// [`super::parse_message_from_completion_text`] reconstruct the original `arguments`. ONE
/// function so the rule is provably the same on the per-key and wrapper paths below.
///
/// The value goes straight into `out` rather than through a `format!` argument: it is the
/// large payload on this path (a file body, a whole search query) and this is the difference
/// between copying it once and twice.
fn push_param(out: &mut String, k: &str, v: &Value) {
    let is_str = matches!(v, Value::String(_));
    out.push_str(&format!(
        "<{DSML}parameter name=\"{k}\" string=\"{is_str}\">"
    ));
    match v {
        Value::String(s) => out.push_str(s),
        other => out.push_str(&python_json(other)),
    }
    out.push_str(&format!("</{DSML}parameter>"));
}

/// `arguments` → one `<｜DSML｜parameter>` per key, `\n`-separated.
fn push_dsml_parameters(out: &mut String, tc: &ToolCall) -> Result<()> {
    // The reference's `try json.loads(...) / except Exception: {"arguments": <the original>}`.
    // The `except` is bare and catches BOTH failure modes: a string that is not JSON, and a
    // non-string that `json.loads` refuses outright (`TypeError`). Either way the ORIGINAL
    // value — not a re-encoding of it — becomes the single `arguments` parameter, so a raw
    // string comes back out with `string="true"` and an object with `string="false"`.
    let parsed = match &tc.arguments {
        Value::String(s) => serde_json::from_str::<Value>(s).ok(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => {
            None
        }
    };
    let Some(parsed) = parsed else {
        push_param(out, "arguments", &tc.arguments);
        return Ok(());
    };
    // `json.loads("5")` SUCCEEDS and then `.items()` raises AttributeError, which is outside
    // the `try`. The reference has no handling for it, so neither is silently invented here.
    let Value::Object(args) = parsed else {
        bail!(
            "tool call `{}` has JSON arguments that parse to {parsed}, which is not an \
             object; the reference raises AttributeError on this",
            tc.name
        )
    };
    for (n, (k, v)) in args.iter().enumerate() {
        if n > 0 {
            out.push('\n');
        }
        push_param(out, k, v);
    }
    Ok(())
}

/// The full DSML tool-call block an assistant turn appends after its content.
fn push_tool_calls(out: &mut String, tool_calls: &[ToolCall]) -> Result<()> {
    out.push_str(&format!("\n\n<{DSML}tool_calls>\n"));
    for (n, tc) in tool_calls.iter().enumerate() {
        if n > 0 {
            out.push('\n');
        }
        push_tool_call(out, tc)?;
    }
    out.push_str(&format!("\n</{DSML}tool_calls>"));
    Ok(())
}

/// Everything that decides how one turn renders and what follows it. Grouped because the
/// three travel together through [`render_turn`] and its two halves; passing them
/// individually was five `usize`/`bool`/`Option` parameters in a row, which is both a
/// CodeScene Primitive Obsession row and a call site where a transposition type-checks.
struct RenderCtx<'a> {
    turns: &'a [Turn],
    opts: &'a EncodeOpts,
    /// The RESOLVED `drop_thinking` (tools already forced it off), not the caller's request.
    drop_thinking: bool,
    last_user: Option<usize>,
}

impl RenderCtx<'_> {
    /// Whether the assistant turn at `index` writes its `<think>` block out.
    ///
    /// Three conditions, named rather than spelled inline at the one call site — the inline
    /// form is a complex conditional the code-health gate scores, and each conjunct is a
    /// separate fact about the reference:
    ///
    /// * thinking mode at all — `Chat` has no reasoning block anywhere;
    /// * the PREVIOUS turn carried no quick-instruction task, because a turn that answers one
    ///   was asked for a token and not for deliberation;
    /// * the turn is at or past the last user message, or `drop_thinking` is off.
    ///
    /// **NOTE the strict `>`** — [`Self::opens_thinking`] uses `>=`. Both are the reference's,
    /// and they differ because this asks "is this a continuation past the last user turn"
    /// while that asks "am I generating for it".
    fn renders_reasoning(&self, index: usize) -> bool {
        if self.opts.thinking != ThinkingMode::Thinking {
            return false;
        }
        let prev_has_task = index
            .checked_sub(1)
            .is_some_and(|p| self.turns.get(p).is_some_and(|t| t.task.is_some()));
        !prev_has_task && (!self.drop_thinking || self.last_user.is_none_or(|l| index > l))
    }

    /// Whether the generation prompt after the turn at `index` leaves `<think>` OPEN.
    ///
    /// The `>=` twin of [`Self::renders_reasoning`]'s `>`; see its note.
    fn opens_thinking(&self, index: usize) -> bool {
        self.opts.thinking == ThinkingMode::Thinking
            && (!self.drop_thinking || self.last_user.is_none_or(|l| index >= l))
    }
}

/// The turn's own bytes — everything before the transition tokens.
fn push_body(out: &mut String, index: usize, ctx: &RenderCtx<'_>, turn: &Turn) -> Result<()> {
    match &turn.body {
        TurnBody::System(i) => {
            out.push_str(&i.content);
            push_instructions_suffix(out, i);
        }
        TurnBody::Developer(i) => {
            // Framed as a user turn — this role has no token of its own.
            ensure!(
                !i.content.is_empty(),
                "a `developer` message must have content (the reference asserts this)"
            );
            out.push_str(USER);
            out.push_str(&i.content);
            push_instructions_suffix(out, i);
        }
        TurnBody::User(blocks) => {
            out.push_str(USER);
            // Appended in place rather than collected and joined: a tool result's body is a
            // whole search result, and the Vec-of-String form copied every one of them twice
            // to produce a string that is immediately pushed here anyway.
            for (n, b) in blocks.iter().enumerate() {
                if n > 0 {
                    out.push_str("\n\n");
                }
                match b {
                    Block::Text(t) => out.push_str(t),
                    Block::ToolResult { content, .. } => {
                        out.push_str("<tool_result>");
                        out.push_str(content);
                        out.push_str("</tool_result>");
                    }
                }
            }
        }
        TurnBody::LatestReminder(content) => {
            out.push_str(LATEST_REMINDER);
            out.push_str(content);
        }
        TurnBody::Assistant(a) => push_assistant(out, index, ctx, a)?,
    }
    Ok(())
}

/// An assistant turn: its reasoning block (when it has one), its content, its tool calls, and
/// its EOS.
///
/// Split out of [`push_body`] because it is the only arm with conditionals of its own, and
/// four of them — the shape the code-health gate scores as a bumpy road. The reference has all
/// five arms in one body.
fn push_assistant(
    out: &mut String,
    index: usize,
    ctx: &RenderCtx<'_>,
    a: &AssistantTurn,
) -> Result<()> {
    if ctx.renders_reasoning(index) {
        out.push_str(&a.reasoning_content);
        out.push_str(THINK_CLOSE);
    }
    out.push_str(&a.content);
    if !a.tool_calls.is_empty() {
        push_tool_calls(out, &a.tool_calls)?;
    }
    if !a.wo_eos {
        out.push_str(EOS);
    }
    Ok(())
}

/// The transition tokens that lead into whatever follows this turn — or nothing, when the
/// next turn frames itself.
fn push_transition(out: &mut String, index: usize, ctx: &RenderCtx<'_>, turn: &Turn) {
    // Only at the end of the conversation or immediately before an assistant /
    // latest_reminder turn. The `latest_reminder` case looks like an oversight in the
    // reference — a user turn followed by a reminder emits `<｜Assistant｜><think>` and *then*
    // the reminder — but it is what the model was trained on, and guessing otherwise is
    // exactly the drift this file exists to prevent.
    let next_turn_frames_itself = ctx
        .turns
        .get(index + 1)
        .is_some_and(|n| !matches!(n.body, TurnBody::Assistant(_) | TurnBody::LatestReminder(_)));
    if next_turn_frames_itself {
        return;
    }

    match turn.task {
        // `action` is the only task that sits after the assistant prefix: it asks the model
        // to route (Search vs Answer) as its first reasoning step.
        Some(Task::Action) => {
            out.push_str(ASSISTANT);
            out.push_str(match ctx.opts.thinking {
                ThinkingMode::Thinking => THINK_OPEN,
                ThinkingMode::Chat => THINK_CLOSE,
            });
            out.push_str(Task::Action.token());
        }
        // Spelled out, not `Some(t)`: a new `Task` that needs Action-style placement (after
        // the assistant prefix rather than after the message) must fail to compile here
        // instead of silently rendering in the wrong position.
        Some(t @ (Task::Query | Task::Authority | Task::Domain | Task::Title | Task::ReadUrl)) => {
            out.push_str(t.token());
        }
        None if turn.is_user_like() => push_generation_prompt(out, index, ctx),
        None => {}
    }
}

/// `<｜Assistant｜>` followed by an OPEN or an immediately-closed `<think>` — the prompt the
/// model generates into.
///
/// One line at its call site, because the `if/else` inside a match arm inside an early-returning
/// function is the second "bump" the code-health gate scores in [`push_transition`], and the
/// decision it makes has a name.
fn push_generation_prompt(out: &mut String, index: usize, ctx: &RenderCtx<'_>) {
    out.push_str(ASSISTANT);
    out.push_str(if ctx.opens_thinking(index) {
        THINK_OPEN
    } else {
        THINK_CLOSE
    });
}

/// Render one turn, including the transition tokens that lead into whatever follows it.
fn render_turn(out: &mut String, index: usize, ctx: &RenderCtx<'_>) -> Result<()> {
    // `get` rather than `[]`: the caller's loop is `0..turns.len()`, so this cannot fire, but
    // the workspace lint table does not deny `indexing_slicing` and a panic in a prompt
    // encoder is a crash in a server. The reference indexes.
    let turn = ctx
        .turns
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("turn {index} is past the conversation"))?;

    // The effort prefix goes before everything, including the system message, and only in
    // thinking mode. `Low` contributes nothing, so this is a no-op for the default.
    if index == 0 && ctx.opts.thinking == ThinkingMode::Thinking {
        out.push_str(ctx.opts.reasoning_effort.prompt());
    }

    push_body(out, index, ctx, turn)?;
    push_transition(out, index, ctx, turn);
    Ok(())
}

/// Encode a conversation into the string the model is prefilled with.
///
/// The result still has to go through the tokenizer; every special token in [`super`] is an
/// entry in `tokenizer.json`'s `added_tokens`, so `Tokenizer::encode` maps them to single ids
/// without any `add_special_tokens` handling —
/// `crates/artifact/tests/v4_encoding_gold.rs::special_tokens_survive_the_tokenizer` is the
/// gate on that claim and carries the argument.
///
/// Errors come from exactly two places, both the reference's own: a `developer` message
/// with no content, and tool-call `arguments` that parse to something other than an object.
pub fn encode_messages(messages: Vec<Message>, opts: &EncodeOpts) -> Result<String> {
    let turns = sort_tool_results_by_call_order(merge_tool_messages(messages));

    // Resolved once. Tool-calling conversations keep every turn's reasoning, because the
    // model has to track multi-step work across the calls.
    let drop_thinking = opts.drop_thinking && !turns.iter().any(Turn::declares_tools);
    let turns = if opts.thinking == ThinkingMode::Thinking && drop_thinking {
        drop_thinking_turns(turns)
    } else {
        turns
    };
    // AFTER the drop: removing a developer turn moves the last user index.
    let ctx = RenderCtx {
        last_user: last_user_index(&turns),
        turns: &turns,
        opts,
        drop_thinking,
    };

    let mut out = String::new();
    if opts.add_bos {
        out.push_str(super::BOS);
    }
    for index in 0..turns.len() {
        // Appended in place. Returning a `String` per turn copied every turn's bytes twice,
        // tool-result bodies included; partial bytes left behind by an `Err` are harmless
        // because `?` throws the whole buffer away.
        render_turn(&mut out, index, &ctx)?;
    }
    Ok(out)
}
