//! HTTP/1.1 on the wire: read one request, write one length-delimited response or one SSE
//! stream. No routing and no JSON vocabulary — those are [`super`] and [`super::oai`].
//!
//! Everything here is generic over `BufRead`/`Write` rather than taking a `TcpStream`,
//! which is what lets the request reader be tested against a byte slice with no socket and
//! no device. That is the whole reason this half is a separate file.

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::io::{BufRead, Write};

#[derive(Debug, PartialEq)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Read one HTTP/1.1 request. `Ok(None)` is a clean EOF — a proxy probing the port.
///
/// Content-Length only: OpenAI clients always send it, and a chunked body arrives here as
/// an empty one and fails as a 400 rather than being half-read into a truncated prompt.
pub fn read_req<R: BufRead>(r: &mut R) -> Result<Option<Req>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    // Strip any query string: `/v1/models?limit=1` is the same route.
    let path = parts
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    ensure!(
        !method.is_empty() && !path.is_empty(),
        "malformed request line {line:?}"
    );
    let len = content_length(r)?;
    let mut body = vec![0u8; len];
    std::io::Read::read_exact(r, &mut body).context("short request body")?;
    Ok(Some(Req { method, path, body }))
}

/// Drain the header block and return the declared body length (0 when absent).
///
/// Split out from [`read_req`] so the loop's one interesting decision — which header
/// matters and how it is spelled — is not buried between the request line and the body
/// read. Header names are case-insensitive and clients disagree about the spelling.
fn content_length<R: BufRead>(r: &mut R) -> Result<usize> {
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((name, v)) = h.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            len = v.trim().parse().context("Content-Length is not a number")?;
        }
    }
    Ok(len)
}

/// One whole JSON response: status line, `Content-Length`, `Connection: close`, body.
/// Length-delimited rather than chunked, which is what makes the keep-alive state
/// machine unnecessary (see the module header in [`super`]).
pub fn send_json(w: &mut impl Write, status: u16, body: &Value) -> Result<()> {
    let body = serde_json::to_vec(body)?;
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        reason_phrase(status),
        body.len(),
    )?;
    write_flush(w, &body)
}

/// Write and flush in one step. Everything this server emits is flushed at the point it
/// is produced — one request per connection, so there is no later write to piggyback on,
/// and an unflushed SSE frame is a token that did not stream.
fn write_flush(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    w.write_all(bytes)?;
    Ok(w.flush()?)
}

/// The SSE preamble — 200 + `text/event-stream`, flushed before the first chunk so the
/// client can start rendering while the first token is still ~a second away.
pub fn sse_head(w: &mut impl Write) -> Result<()> {
    write_flush(
        w,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Connection: close\r\n\r\n",
    )
}

/// One `data:` event. Returns the write error rather than logging it — the caller
/// reads a failure here as the client having hung up and stops the decode.
pub fn sse(w: &mut impl Write, v: &Value) -> Result<()> {
    write!(w, "data: {}\n\n", serde_json::to_string(v)?)?;
    w.flush()?; // per event: an unflushed token is a token that did not stream
    Ok(())
}

/// The terminator every SSE consumer waits for before it stops reading.
pub fn sse_done(w: &mut impl Write) -> Result<()> {
    write_flush(w, b"data: [DONE]\n\n")
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a test harness that cannot parse its own fixture should panic
mod tests {
    use super::*;
    use std::io::BufReader;

    fn req(bytes: &str) -> Option<Req> {
        read_req(&mut BufReader::new(bytes.as_bytes())).unwrap()
    }

    #[test]
    fn reads_a_post_with_a_body() {
        let r = req(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 7\r\n\r\n{\"a\":1}",
        )
        .unwrap();
        assert_eq!(r.method, "POST");
        assert_eq!(r.path, "/v1/chat/completions");
        assert_eq!(r.body, b"{\"a\":1}");
    }

    #[test]
    fn header_case_and_query_string_do_not_change_the_route() {
        let r = req("GET /v1/models?limit=1 HTTP/1.1\r\ncontent-length: 0\r\n\r\n").unwrap();
        assert_eq!(r.path, "/v1/models");
        assert!(r.body.is_empty());
    }

    #[test]
    fn clean_eof_is_not_an_error() {
        assert_eq!(req(""), None);
    }

    #[test]
    fn a_short_body_fails_rather_than_serving_a_truncated_prompt() {
        // Content-Length lies: 99 promised, 7 delivered. Reading what arrived would
        // silently answer a conversation the client did not send.
        assert!(
            read_req(&mut BufReader::new(
                "POST /x HTTP/1.1\r\nContent-Length: 99\r\n\r\n{\"a\":1}".as_bytes()
            ))
            .is_err()
        );
    }

    /// The deviceless smoke of the response half: a client parses the framing, so the
    /// framing is part of the contract, not an implementation detail. A `Content-Length`
    /// that disagreed with the body is the one failure here a client reports as a hang.
    #[test]
    fn a_json_response_is_length_delimited_and_closes_the_connection() {
        let mut out: Vec<u8> = Vec::new();
        send_json(&mut out, 400, &serde_json::json!({"a": 1})).unwrap();
        let text = String::from_utf8(out).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 400 Bad Request\r\n"), "{head}");
        assert!(head.contains("Connection: close"), "{head}");
        assert!(
            head.contains(&format!("Content-Length: {}", body.len())),
            "declared length disagrees with the {} body bytes:\n{head}",
            body.len()
        );
        assert_eq!(body, "{\"a\":1}");
    }

    /// SSE frames are `data: <json>\n\n` and nothing else — a client splits on the blank
    /// line, so a stray `\r` or a missing terminator strands the last token.
    #[test]
    fn sse_frames_are_blank_line_terminated_and_end_with_done() {
        let mut out: Vec<u8> = Vec::new();
        sse(&mut out, &serde_json::json!({"delta": "hi"})).unwrap();
        sse_done(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "data: {\"delta\":\"hi\"}\n\ndata: [DONE]\n\n"
        );
    }
}
