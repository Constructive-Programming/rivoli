//! Minimal JSON reader — one flat parse for safetensors headers and
//! config.json, without pulling serde_json.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub enum Value {
    Object(BTreeMap<String, Value>),
    Array(Vec<Value>),
    Str(String),
    Num(f64),
    Bool,
    Null,
}

impl Value {
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Num(n) => Some(*n as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.as_object().and_then(|m| m.get(k))
    }
}

pub fn parse(s: &str) -> Result<Value> {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    p.ws();
    p.value()
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Result<Value> {
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.lit("true", Value::Bool),
            Some(b'f') => self.lit("false", Value::Bool),
            Some(b'n') => self.lit("null", Value::Null),
            Some(_) => self.number(),
            None => bail!("unexpected end of json"),
        }
    }
    fn lit(&mut self, kw: &str, v: Value) -> Result<Value> {
        if self.b[self.i..].starts_with(kw.as_bytes()) {
            self.i += kw.len();
            Ok(v)
        } else {
            bail!("bad literal at {}", self.i)
        }
    }
    fn object(&mut self) -> Result<Value> {
        self.i += 1; // {
        let mut m = BTreeMap::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::Object(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                bail!("expected ':' at {}", self.i);
            }
            self.i += 1;
            let v = self.value()?;
            m.insert(k, v);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => bail!("expected ',' or '}}' at {}", self.i),
            }
        }
        Ok(Value::Object(m))
    }
    fn array(&mut self) -> Result<Value> {
        self.i += 1; // [
        let mut a = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::Array(a));
        }
        loop {
            a.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => bail!("expected ',' or ']' at {}", self.i),
            }
        }
        Ok(Value::Array(a))
    }
    fn string(&mut self) -> Result<String> {
        if self.b.get(self.i) != Some(&b'"') {
            bail!("expected '\"' at {}", self.i);
        }
        self.i += 1;
        let mut out = String::new();
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).unwrap_or(&b'"');
                    self.i += 1;
                    out.push(match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        other => other as char,
                    });
                }
                _ => out.push(c as char),
            }
        }
        bail!("unterminated string")
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.i;
        while let Some(&c) = self.b.get(self.i) {
            if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                self.i += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i])?;
        Ok(Value::Num(s.parse()?))
    }
}
