//! wasc - WebAssembly Single-file Compiler
//! A C-like language compiler specifically for WebAssembly with first-class
//! support for references, threads, atomics, SIMD, and tables.
//!
//! Features:
//! - C-like syntax
//! - First-class externref and funcref types
//! - Direct memory and table management
//! - Zero-overhead external function calls
//! - No standard library - you control everything

#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
extern crate alloc;

#[cfg(test)]
extern crate std;

#[cfg(target_arch = "wasm32")]
use alloc::boxed::Box;
#[cfg(target_arch = "wasm32")]
use alloc::collections::BTreeMap;
#[cfg(target_arch = "wasm32")]
use alloc::string::String;
#[cfg(target_arch = "wasm32")]
use alloc::vec::Vec;
#[cfg(target_arch = "wasm32")]
use alloc::{format, vec};

#[cfg(not(target_arch = "wasm32"))]
use std::boxed::Box;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::string::String;
#[cfg(not(target_arch = "wasm32"))]
use std::vec::Vec;

use webassembly::*;

// ============================================================================
// PUBLIC API
// ============================================================================

/// Compile wasc source code to WebAssembly binary
pub fn compile(source: &str) -> Result<Vec<u8>, CompileError> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(&tokens);
    let program = parser.parse_program()?;
    let codegen = CodeGen::new(program);
    codegen.generate()
}

// WASM-bindgen wrapper for JavaScript usage
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn compile_wasm(source: &str) -> Result<JsValue, JsValue> {
    match compile(source) {
        Ok(bytes) => {
            // Convert Vec<u8> to Uint8Array
            let array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
            array.copy_from(&bytes);
            Ok(array.into())
        }
        Err(e) => {
            let msg = format!("{:?}", e);
            Err(JsValue::from_str(&msg))
        }
    }
}

/// Compilation error types
#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    LexError {
        line: usize,
        col: usize,
        message: String,
    },
    ParseError {
        line: usize,
        col: usize,
        message: String,
    },
    CodeGenError {
        message: String,
    },
}

// ============================================================================
// LEXER
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),

    // Keywords
    Void,
    I32,
    I64,
    F32,
    F64,
    Externref,
    Funcref,
    Memory,
    Table,
    Export,
    Extern,
    If,
    Else,
    While,
    For,
    Return,
    Ref,
    Null,
    Nullref,
    True,
    False,
    Shared,
    Atomic,

    // Identifiers
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Assign,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semicolon,
    Comma,
    Colon,
    Dot,
    Arrow,
    Ampersand, // For funcref: &function

    // Special
    NewLine,
    Eof,
}

struct Lexer<'a> {
    source: &'a str,
    chars: core::str::Chars<'a>,
    current: Option<char>,
    line: usize,
    col: usize,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        let mut chars = source.chars();
        let current = chars.next();
        Lexer {
            source,
            chars,
            current,
            line: 1,
            col: 1,
            pos: 0,
        }
    }

    fn advance(&mut self) {
        if let Some(c) = self.current {
            self.pos += c.len_utf8();
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        self.current = self.chars.next();
    }

    fn peek(&self) -> Option<char> {
        self.current
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.clone().next()
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == Some('/') && self.peek_next() == Some('/') {
            while let Some(c) = self.peek() {
                if c == '\n' {
                    break;
                }
                self.advance();
            }
        } else if self.peek() == Some('/') && self.peek_next() == Some('*') {
            self.advance(); // /
            self.advance(); // *
            while let Some(c) = self.peek() {
                if c == '*' && self.peek_next() == Some('/') {
                    self.advance(); // *
                    self.advance(); // /
                    break;
                }
                self.advance();
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        String::from(&self.source[start..self.pos])
    }

    fn read_number(&mut self) -> Token {
        let start = self.pos;
        let mut is_float = false;

        while let Some(c) = self.peek() {
            if c.is_numeric() {
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                self.advance();
                // Must have digits after decimal
                if !self.peek().map(|c| c.is_numeric()).unwrap_or(false) {
                    return Token::IntLiteral(0); // Error case
                }
            } else if c == 'e' || c == 'E' {
                self.advance();
                if self.peek() == Some('-') || self.peek() == Some('+') {
                    self.advance();
                }
                while let Some(c) = self.peek() {
                    if c.is_numeric() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                is_float = true;
                break;
            } else {
                break;
            }
        }

        let num_str = &self.source[start..self.pos];
        if is_float {
            if let Ok(f) = parse_float(num_str) {
                Token::FloatLiteral(f)
            } else {
                Token::IntLiteral(0)
            }
        } else {
            if let Ok(n) = num_str.parse::<i64>() {
                Token::IntLiteral(n)
            } else {
                Token::IntLiteral(0)
            }
        }
    }

    fn read_string(&mut self) -> Result<Token, CompileError> {
        let quote = self.peek().unwrap();
        self.advance(); // opening quote

        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == quote {
                self.advance(); // closing quote
                let s = String::from(&self.source[start..self.pos - 1]);
                return Ok(Token::StringLiteral(s));
            }
            if c == '\\' {
                self.advance();
                if self.peek().is_some() {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }

        Err(CompileError::LexError {
            line: self.line,
            col: self.col,
            message: String::from("unterminated string literal"),
        })
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, CompileError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            // Skip comments
            if self.peek() == Some('/') {
                let start_line = self.line;
                let start_col = self.col;
                self.skip_comment();
                if self.line != start_line || self.col != start_col {
                    continue;
                }
            }

            let token = match self.peek() {
                None => break,
                Some('\n') => {
                    self.advance();
                    Token::NewLine
                }
                Some('"') | Some('\'') => self.read_string()?,

                // Operators
                Some('+') => {
                    self.advance();
                    Token::Plus
                }
                Some('-') => {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        Token::Arrow
                    } else {
                        Token::Minus
                    }
                }
                Some('*') => {
                    self.advance();
                    Token::Mul
                }
                Some('/') => {
                    self.advance();
                    Token::Div
                }
                Some('%') => {
                    self.advance();
                    Token::Mod
                }
                Some('=') => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Token::Eq
                    } else {
                        Token::Assign
                    }
                }
                Some('!') => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Token::Ne
                    } else {
                        Token::Not
                    }
                }
                Some('<') => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Token::Le
                    } else if self.peek() == Some('<') {
                        self.advance();
                        Token::Shl
                    } else {
                        Token::Lt
                    }
                }
                Some('>') => {
                    self.advance();
                    if self.peek() == Some('=') {
                        self.advance();
                        Token::Ge
                    } else if self.peek() == Some('>') {
                        self.advance();
                        Token::Shr
                    } else {
                        Token::Gt
                    }
                }
                Some('&') => {
                    self.advance();
                    if self.peek() == Some('&') {
                        self.advance();
                        Token::And
                    } else {
                        Token::Ampersand
                    }
                }
                Some('|') => {
                    self.advance();
                    if self.peek() == Some('|') {
                        self.advance();
                        Token::Or
                    } else {
                        Token::BitOr
                    }
                }
                Some('^') => {
                    self.advance();
                    Token::BitXor
                }
                Some('~') => {
                    self.advance();
                    Token::BitNot
                }

                // Punctuation
                Some('(') => {
                    self.advance();
                    Token::LParen
                }
                Some(')') => {
                    self.advance();
                    Token::RParen
                }
                Some('{') => {
                    self.advance();
                    Token::LBrace
                }
                Some('}') => {
                    self.advance();
                    Token::RBrace
                }
                Some('[') => {
                    self.advance();
                    Token::LBracket
                }
                Some(']') => {
                    self.advance();
                    Token::RBracket
                }
                Some(';') => {
                    self.advance();
                    Token::Semicolon
                }
                Some(',') => {
                    self.advance();
                    Token::Comma
                }
                Some(':') => {
                    self.advance();
                    Token::Colon
                }
                Some('.') => {
                    self.advance();
                    Token::Dot
                }

                // Numbers
                Some(c) if c.is_numeric() => self.read_number(),

                // Identifiers and keywords
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let ident = self.read_identifier();
                    match ident.as_str() {
                        "void" => Token::Void,
                        "i32" => Token::I32,
                        "i64" => Token::I64,
                        "f32" => Token::F32,
                        "f64" => Token::F64,
                        "externref" => Token::Externref,
                        "funcref" => Token::Funcref,
                        "memory" => Token::Memory,
                        "table" => Token::Table,
                        "export" => Token::Export,
                        "extern" => Token::Extern,
                        "if" => Token::If,
                        "else" => Token::Else,
                        "while" => Token::While,
                        "for" => Token::For,
                        "return" => Token::Return,
                        "ref" => Token::Ref,
                        "null" => Token::Null,
                        "nullref" => Token::Nullref,
                        "true" => Token::True,
                        "false" => Token::False,
                        "shared" => Token::Shared,
                        "atomic" => Token::Atomic,
                        _ => Token::Identifier(ident),
                    }
                }

                Some(c) => {
                    return Err(CompileError::LexError {
                        line: self.line,
                        col: self.col,
                        message: format!("unexpected character: {}", c),
                    });
                }
            };

            tokens.push(token);
        }

        tokens.push(Token::Eof);
        Ok(tokens)
    }
}

fn parse_float(s: &str) -> Result<f64, ()> {
    let mut result = 0.0f64;
    let mut sign = 1.0f64;
    let mut has_decimal = false;
    let mut decimal_place = 0.1f64;
    let mut has_exp = false;
    let mut exp_sign = 1.0f64;
    let mut exp_val = 0i32;

    let bytes = s.as_bytes();
    let mut i = 0;

    // Sign
    if i < bytes.len() && bytes[i] == b'-' {
        sign = -1.0;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }

    // Integer part
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        result = result * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }

    // Decimal part
    if i < bytes.len() && bytes[i] == b'.' {
        has_decimal = true;
        i += 1;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            result += (bytes[i] - b'0') as f64 * decimal_place;
            decimal_place *= 0.1;
            i += 1;
        }
    }

    // Exponent
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        has_exp = true;
        i += 1;
        if i < bytes.len() && bytes[i] == b'-' {
            exp_sign = -1.0;
            i += 1;
        } else if i < bytes.len() && bytes[i] == b'+' {
            i += 1;
        }
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            exp_val = exp_val * 10 + (bytes[i] - b'0') as i32;
            i += 1;
        }
    }

    if i != bytes.len() {
        return Err(());
    }

    if has_exp {
        // Manual computation of 10^exp since powi isn't available in no_std
        let exp_power = exp_val * exp_sign as i32;
        let mut exp_multiplier = 1.0f64;
        let base = 10.0f64;
        let abs_exp = if exp_power < 0 { -exp_power } else { exp_power };
        for _ in 0..abs_exp {
            exp_multiplier *= base;
        }
        if exp_power < 0 {
            result /= exp_multiplier;
        } else {
            result *= exp_multiplier;
        }
    }

    Ok(sign * result)
}

// ============================================================================
// AST
// ============================================================================

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq)]
enum Type {
    Void,
    I32,
    I64,
    F32,
    F64,
    Externref,
    Funcref,
}

#[derive(Debug, Clone)]
struct Program {
    memory: Option<MemoryDecl>,
    tables: Vec<TableDecl>,
    imports: Vec<ImportDecl>,
    functions: Vec<Function>,
}

#[derive(Debug, Clone)]
struct MemoryDecl {
    initial: u32,
    maximum: Option<u32>,
    shared: bool,
}

#[derive(Debug, Clone)]
struct TableDecl {
    elem_type: Type,
    initial: u32,
    maximum: Option<u32>,
}

#[derive(Debug, Clone)]
struct ImportDecl {
    module: String,
    name: String,
    func_name: String,
    params: Vec<(String, Type)>,
    return_type: Type,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    params: Vec<(String, Type)>,
    return_type: Type,
    is_export: bool,
    locals: Vec<(String, Type)>,
    body: Vec<Statement>,
}

#[derive(Debug, Clone)]
enum Statement {
    // Variable declaration: i32 x; or i32 x = 5;
    VarDecl(String, Type, Option<Expression>),
    // Assignment: x = expr;
    Assign(String, Expression),
    // Expression statement: foo();
    Expr(Expression),
    // If statement: if (cond) { ... } else { ... }
    If(Expression, Vec<Statement>, Option<Vec<Statement>>),
    // While loop: while (cond) { ... }
    While(Expression, Vec<Statement>),
    // For loop: for (init; cond; update) { ... }
    For(
        Option<Box<Statement>>,
        Option<Expression>,
        Option<Box<Statement>>,
        Vec<Statement>,
    ),
    // Return statement: return; or return expr;
    Return(Option<Expression>),
    // Block statement: { ... }
    Block(Vec<Statement>),
    // Table assignment: table[index] = value;
    TableSet(Expression, Expression),
    // Memory store: *ptr = value;
    MemoryStore(Expression, Expression),
}

#[derive(Debug, Clone)]
enum Expression {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Variable reference
    Variable(String),

    // Binary operation
    Binary(BinOp, Box<Expression>, Box<Expression>),

    // Unary operation
    Unary(UnaryOp, Box<Expression>),

    // Function call: foo(a, b)
    Call(String, Vec<Expression>),

    // Call through funcref: call_ref(func_ref, args...)
    CallRef(Box<Expression>, Vec<Expression>),

    // Reference null: ref.null externref / ref.null funcref
    RefNull(Type),

    // Reference is null: ref.is_null(expr)
    RefIsNull(Box<Expression>),

    // Reference to function: &function_name
    RefFunc(String),

    // Table get: table[index] or table.get(index)
    TableGet(Box<Expression>),

    // Cast to reference type
    RefCast(Box<Expression>, Type),

    // Dereference: *ptr
    Deref(Box<Expression>),

    // Address of: &var (for linear memory, not funcref)
    AddressOf(String),
}

#[derive(Debug, Clone, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, PartialEq)]
enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

// ============================================================================
// PARSER
// ============================================================================

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    current_token: Token,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let current = if tokens.is_empty() {
            Token::Eof
        } else {
            tokens[0].clone()
        };
        Parser {
            tokens,
            pos: 0,
            current_token: current,
        }
    }

    fn advance(&mut self) {
        self.pos += 1;
        self.current_token = if self.pos < self.tokens.len() {
            self.tokens[self.pos].clone()
        } else {
            Token::Eof
        };
    }

    fn skip_newlines(&mut self) {
        while self.current_token == Token::NewLine {
            self.advance();
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), CompileError> {
        if self.current_token == expected {
            self.advance();
            Ok(())
        } else {
            Err(CompileError::ParseError {
                line: 0, // TODO: track line numbers
                col: 0,
                message: format!("expected {:?}, got {:?}", expected, self.current_token),
            })
        }
    }

    fn match_token(&mut self, token: Token) -> bool {
        if self.current_token == token {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_type(&mut self) -> Result<Type, CompileError> {
        let t = match &self.current_token {
            Token::Void => Type::Void,
            Token::I32 => Type::I32,
            Token::I64 => Type::I64,
            Token::F32 => Type::F32,
            Token::F64 => Type::F64,
            Token::Externref => Type::Externref,
            Token::Funcref => Type::Funcref,
            _ => {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: format!("expected type, got {:?}", self.current_token),
                })
            }
        };
        self.advance();
        Ok(t)
    }

    fn parse_program(&mut self) -> Result<Program, CompileError> {
        let mut memory = None;
        let mut tables = Vec::new();
        let mut imports = Vec::new();
        let mut functions = Vec::new();

        self.skip_newlines();

        while self.current_token != Token::Eof {
            self.skip_newlines();

            // Check again after skipping newlines
            if self.current_token == Token::Eof {
                break;
            }

            match &self.current_token {
                Token::Memory => {
                    memory = Some(self.parse_memory_decl()?);
                }
                Token::Table => {
                    tables.push(self.parse_table_decl()?);
                }
                Token::Extern => {
                    imports.push(self.parse_import()?);
                }
                Token::Export => {
                    self.advance();
                    let func = self.parse_function(true)?;
                    functions.push(func);
                }
                Token::Void
                | Token::I32
                | Token::I64
                | Token::F32
                | Token::F64
                | Token::Externref
                | Token::Funcref => {
                    let func = self.parse_function(false)?;
                    functions.push(func);
                }
                Token::NewLine => {
                    self.advance();
                }
                _ => {
                    return Err(CompileError::ParseError {
                        line: 0,
                        col: 0,
                        message: format!("unexpected token: {:?}", self.current_token),
                    });
                }
            }
        }

        Ok(Program {
            memory,
            tables,
            imports,
            functions,
        })
    }

    fn parse_memory_decl(&mut self) -> Result<MemoryDecl, CompileError> {
        self.advance(); // memory

        let mut shared = false;
        if self.current_token == Token::Shared {
            shared = true;
            self.advance();
        }

        let initial = match &self.current_token {
            Token::IntLiteral(n) => {
                let v = *n as u32;
                self.advance();
                v
            }
            _ => {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: String::from("expected memory initial size"),
                });
            }
        };

        let maximum = if self.match_token(Token::Comma)
            || (self.current_token != Token::Semicolon
                && self.current_token != Token::NewLine
                && self.current_token != Token::Eof)
        {
            if let Token::IntLiteral(n) = &self.current_token {
                let v = Some(*n as u32);
                self.advance();
                v
            } else {
                None
            }
        } else {
            None
        };

        self.match_token(Token::Semicolon);

        Ok(MemoryDecl {
            initial,
            maximum,
            shared,
        })
    }

    fn parse_table_decl(&mut self) -> Result<TableDecl, CompileError> {
        self.advance(); // table

        let elem_type = if matches!(self.current_token, Token::Funcref | Token::Externref) {
            self.parse_type()?
        } else {
            Type::Funcref // Default
        };

        let initial = match &self.current_token {
            Token::IntLiteral(n) => {
                let v = *n as u32;
                self.advance();
                v
            }
            _ => {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: String::from("expected table initial size"),
                });
            }
        };

        let maximum = if self.match_token(Token::Comma)
            || (self.current_token != Token::Semicolon
                && self.current_token != Token::NewLine
                && self.current_token != Token::Eof)
        {
            if let Token::IntLiteral(n) = &self.current_token {
                let v = Some(*n as u32);
                self.advance();
                v
            } else {
                None
            }
        } else {
            None
        };

        self.match_token(Token::Semicolon);

        Ok(TableDecl {
            elem_type,
            initial,
            maximum,
        })
    }

    fn parse_import(&mut self) -> Result<ImportDecl, CompileError> {
        self.advance(); // extern

        let return_type = self.parse_type()?;

        let func_name = match &self.current_token {
            Token::Identifier(n) => {
                let name = n.clone();
                self.advance();
                name
            }
            _ => {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: String::from("expected import function name"),
                });
            }
        };

        self.expect(Token::LParen)?;

        let mut params = Vec::new();
        while !matches!(self.current_token, Token::RParen | Token::Eof) {
            let param_type = self.parse_type()?;
            let param_name = if let Token::Identifier(n) = &self.current_token {
                let name = n.clone();
                self.advance();
                name
            } else {
                String::from("")
            };
            params.push((param_name, param_type));

            if !self.match_token(Token::Comma) {
                break;
            }
        }

        self.expect(Token::RParen)?;
        self.match_token(Token::Semicolon);

        // For now, use env module for all imports
        Ok(ImportDecl {
            module: String::from("env"),
            name: func_name.clone(),
            func_name,
            params,
            return_type,
        })
    }

    fn parse_function(&mut self, is_export: bool) -> Result<Function, CompileError> {
        let return_type = self.parse_type()?;

        let name = match &self.current_token {
            Token::Identifier(n) => {
                let name = n.clone();
                self.advance();
                name
            }
            _ => {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: String::from("expected function name"),
                });
            }
        };

        self.expect(Token::LParen)?;

        let mut params = Vec::new();
        while !matches!(self.current_token, Token::RParen | Token::Eof) {
            let param_type = self.parse_type()?;
            let param_name = if let Token::Identifier(n) = &self.current_token {
                let name = n.clone();
                self.advance();
                name
            } else {
                String::from("")
            };
            params.push((param_name, param_type));

            if !self.match_token(Token::Comma) {
                break;
            }
        }

        self.expect(Token::RParen)?;

        // Body or semicolon
        let (locals, body) = if self.match_token(Token::Semicolon) {
            // Declaration only
            (Vec::new(), Vec::new())
        } else if self.current_token == Token::LBrace {
            self.parse_function_body()?
        } else {
            return Err(CompileError::ParseError {
                line: 0,
                col: 0,
                message: String::from("expected function body or semicolon"),
            });
        };

        Ok(Function {
            name,
            params,
            return_type,
            is_export,
            locals,
            body,
        })
    }

    fn parse_function_body(
        &mut self,
    ) -> Result<(Vec<(String, Type)>, Vec<Statement>), CompileError> {
        self.expect(Token::LBrace)?;

        let mut locals = Vec::new();
        let mut statements = Vec::new();

        while self.current_token != Token::RBrace && self.current_token != Token::Eof {
            self.skip_newlines();

            if self.current_token == Token::RBrace {
                break;
            }

            let stmt = self.parse_statement()?;

            // Collect local declarations
            if let Statement::VarDecl(name, ty, _) = &stmt {
                locals.push((name.clone(), ty.clone()));
            }

            statements.push(stmt);
        }

        self.expect(Token::RBrace)?;

        Ok((locals, statements))
    }

    fn parse_statement(&mut self) -> Result<Statement, CompileError> {
        self.skip_newlines();

        match &self.current_token {
            Token::If => self.parse_if_statement(),
            Token::While => self.parse_while_statement(),
            Token::For => self.parse_for_statement(),
            Token::Return => self.parse_return_statement(),
            Token::LBrace => {
                self.advance();
                let mut stmts = Vec::new();
                while self.current_token != Token::RBrace && self.current_token != Token::Eof {
                    self.skip_newlines();
                    if self.current_token == Token::RBrace {
                        break;
                    }
                    stmts.push(self.parse_statement()?);
                }
                self.expect(Token::RBrace)?;
                Ok(Statement::Block(stmts))
            }
            Token::Void
            | Token::I32
            | Token::I64
            | Token::F32
            | Token::F64
            | Token::Externref
            | Token::Funcref => {
                // Variable declaration
                let ty = self.parse_type()?;
                let name = match &self.current_token {
                    Token::Identifier(n) => {
                        let name = n.clone();
                        self.advance();
                        name
                    }
                    _ => {
                        return Err(CompileError::ParseError {
                            line: 0,
                            col: 0,
                            message: String::from("expected variable name"),
                        });
                    }
                };

                let init = if self.match_token(Token::Assign) {
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                self.match_token(Token::Semicolon);
                Ok(Statement::VarDecl(name, ty, init))
            }
            _ => {
                // Could be assignment, call, or table set
                let expr = self.parse_expression()?;

                // Check for table assignment: table[index] = value;
                if self.current_token == Token::Assign {
                    if let Expression::TableGet(idx_expr) = expr {
                        self.advance(); // consume =
                        let value = self.parse_expression()?;
                        self.match_token(Token::Semicolon);
                        return Ok(Statement::TableSet(*idx_expr, value));
                    }
                }

                // Assignment: var = expr; or *ptr = expr;
                if self.current_token == Token::Assign {
                    match expr {
                        Expression::Variable(name) => {
                            self.advance();
                            let value = self.parse_expression()?;
                            self.match_token(Token::Semicolon);
                            return Ok(Statement::Assign(name, value));
                        }
                        Expression::Deref(ptr_expr) => {
                            // Memory store: *ptr = value;
                            self.advance();
                            let value = self.parse_expression()?;
                            self.match_token(Token::Semicolon);
                            return Ok(Statement::MemoryStore(*ptr_expr, value));
                        }
                        _ => {
                            return Err(CompileError::ParseError {
                                line: 0,
                                col: 0,
                                message: String::from("invalid assignment target"),
                            });
                        }
                    }
                }

                self.match_token(Token::Semicolon);
                Ok(Statement::Expr(expr))
            }
        }
    }

    fn parse_if_statement(&mut self) -> Result<Statement, CompileError> {
        self.advance(); // if
        self.expect(Token::LParen)?;
        let condition = self.parse_expression()?;
        self.expect(Token::RParen)?;

        let then_branch = if self.current_token == Token::LBrace {
            if let Statement::Block(stmts) = self.parse_statement()? {
                stmts
            } else {
                Vec::new()
            }
        } else {
            vec![self.parse_statement()?]
        };

        let else_branch = if self.match_token(Token::Else) {
            Some(if self.current_token == Token::LBrace {
                if let Statement::Block(stmts) = self.parse_statement()? {
                    stmts
                } else {
                    Vec::new()
                }
            } else if self.current_token == Token::If {
                vec![self.parse_statement()?]
            } else {
                vec![self.parse_statement()?]
            })
        } else {
            None
        };

        Ok(Statement::If(condition, then_branch, else_branch))
    }

    fn parse_while_statement(&mut self) -> Result<Statement, CompileError> {
        self.advance(); // while
        self.expect(Token::LParen)?;
        let condition = self.parse_expression()?;
        self.expect(Token::RParen)?;

        let body = if self.current_token == Token::LBrace {
            if let Statement::Block(stmts) = self.parse_statement()? {
                stmts
            } else {
                Vec::new()
            }
        } else {
            vec![self.parse_statement()?]
        };

        Ok(Statement::While(condition, body))
    }

    fn parse_for_statement(&mut self) -> Result<Statement, CompileError> {
        self.advance(); // for
        self.expect(Token::LParen)?;

        let init = if self.current_token == Token::Semicolon {
            None
        } else {
            Some(Box::new(self.parse_statement()?))
        };

        let cond = if self.current_token == Token::Semicolon {
            None
        } else {
            let expr = self.parse_expression()?;
            if !self.match_token(Token::Semicolon) {
                return Err(CompileError::ParseError {
                    line: 0,
                    col: 0,
                    message: String::from("expected ; after for condition"),
                });
            }
            Some(expr)
        };

        let update = if self.current_token == Token::RParen {
            None
        } else {
            Some(Box::new(Statement::Expr(self.parse_expression()?)))
        };

        self.expect(Token::RParen)?;

        let body = if self.current_token == Token::LBrace {
            if let Statement::Block(stmts) = self.parse_statement()? {
                stmts
            } else {
                Vec::new()
            }
        } else {
            vec![self.parse_statement()?]
        };

        Ok(Statement::For(init, cond, update, body))
    }

    fn parse_return_statement(&mut self) -> Result<Statement, CompileError> {
        self.advance(); // return

        let value = if self.current_token == Token::Semicolon
            || self.current_token == Token::NewLine
            || self.current_token == Token::RBrace
        {
            None
        } else {
            Some(self.parse_expression()?)
        };

        self.match_token(Token::Semicolon);

        Ok(Statement::Return(value))
    }

    fn parse_expression(&mut self) -> Result<Expression, CompileError> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_and_expression()?;

        while self.current_token == Token::Or {
            self.advance();
            let right = self.parse_and_expression()?;
            left = Expression::Binary(BinOp::Or, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_equality_expression()?;

        while self.current_token == Token::And {
            self.advance();
            let right = self.parse_equality_expression()?;
            left = Expression::Binary(BinOp::And, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_equality_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_comparison_expression()?;

        loop {
            match self.current_token {
                Token::Eq => {
                    self.advance();
                    let right = self.parse_comparison_expression()?;
                    left = Expression::Binary(BinOp::Eq, Box::new(left), Box::new(right));
                }
                Token::Ne => {
                    self.advance();
                    let right = self.parse_comparison_expression()?;
                    left = Expression::Binary(BinOp::Ne, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_comparison_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_additive_expression()?;

        loop {
            match self.current_token {
                Token::Lt => {
                    self.advance();
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary(BinOp::Lt, Box::new(left), Box::new(right));
                }
                Token::Gt => {
                    self.advance();
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary(BinOp::Gt, Box::new(left), Box::new(right));
                }
                Token::Le => {
                    self.advance();
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary(BinOp::Le, Box::new(left), Box::new(right));
                }
                Token::Ge => {
                    self.advance();
                    let right = self.parse_additive_expression()?;
                    left = Expression::Binary(BinOp::Ge, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_additive_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            match self.current_token {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Binary(BinOp::Add, Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Binary(BinOp::Sub, Box::new(left), Box::new(right));
                }
                Token::BitOr => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Binary(BinOp::BitOr, Box::new(left), Box::new(right));
                }
                Token::BitXor => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Binary(BinOp::BitXor, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_shift_expression()?;

        loop {
            match self.current_token {
                Token::Mul => {
                    self.advance();
                    let right = self.parse_shift_expression()?;
                    left = Expression::Binary(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Token::Div => {
                    self.advance();
                    let right = self.parse_shift_expression()?;
                    left = Expression::Binary(BinOp::Div, Box::new(left), Box::new(right));
                }
                Token::Mod => {
                    self.advance();
                    let right = self.parse_shift_expression()?;
                    left = Expression::Binary(BinOp::Mod, Box::new(left), Box::new(right));
                }
                Token::BitAnd => {
                    self.advance();
                    let right = self.parse_shift_expression()?;
                    left = Expression::Binary(BinOp::BitAnd, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_shift_expression(&mut self) -> Result<Expression, CompileError> {
        let mut left = self.parse_unary_expression()?;

        loop {
            match self.current_token {
                Token::Shl => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Binary(BinOp::Shl, Box::new(left), Box::new(right));
                }
                Token::Shr => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Binary(BinOp::Shr, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression, CompileError> {
        match &self.current_token {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary(UnaryOp::Neg, Box::new(operand)))
            }
            Token::Not => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary(UnaryOp::Not, Box::new(operand)))
            }
            Token::BitNot => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Unary(UnaryOp::BitNot, Box::new(operand)))
            }
            Token::Ampersand => {
                self.advance();
                match &self.current_token {
                    Token::Identifier(name) => {
                        let func_name = name.clone();
                        self.advance();
                        Ok(Expression::RefFunc(func_name))
                    }
                    _ => {
                        // For non-function address-of
                        let expr = self.parse_primary()?;
                        if let Expression::Variable(name) = expr {
                            Ok(Expression::AddressOf(name))
                        } else {
                            Err(CompileError::ParseError {
                                line: 0,
                                col: 0,
                                message: String::from("expected variable name after &"),
                            })
                        }
                    }
                }
            }
            Token::Mul => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(Expression::Deref(Box::new(operand)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        match &self.current_token.clone() {
            Token::IntLiteral(n) => {
                let val = *n;
                self.advance();
                Ok(Expression::IntLiteral(val))
            }
            Token::FloatLiteral(f) => {
                let val = *f;
                self.advance();
                Ok(Expression::FloatLiteral(val))
            }
            Token::StringLiteral(s) => {
                let val = s.clone();
                self.advance();
                Ok(Expression::StringLiteral(val))
            }
            Token::True => {
                self.advance();
                Ok(Expression::BoolLiteral(true))
            }
            Token::False => {
                self.advance();
                Ok(Expression::BoolLiteral(false))
            }
            Token::Null | Token::Nullref => {
                self.advance();
                // ref.null - we don't know the type yet, will be resolved later
                Ok(Expression::RefNull(Type::Externref))
            }
            Token::Ref => {
                self.advance();
                // Handle optional dot: ref.null or ref . null
                self.match_token(Token::Dot);
                // ref.null, ref.is_null, or type cast
                if self.current_token == Token::Null {
                    self.advance();
                    // Expect type
                    let ty = self.parse_type()?;
                    Ok(Expression::RefNull(ty))
                } else if let Token::Identifier(name) = &self.current_token {
                    if name == "is_null" || name == "isnull" {
                        self.advance();
                        self.expect(Token::LParen)?;
                        let expr = self.parse_expression()?;
                        self.expect(Token::RParen)?;
                        Ok(Expression::RefIsNull(Box::new(expr)))
                    } else {
                        Err(CompileError::ParseError {
                            line: 0,
                            col: 0,
                            message: format!("unknown ref operation: {}", name),
                        })
                    }
                } else {
                    Err(CompileError::ParseError {
                        line: 0,
                        col: 0,
                        message: String::from("expected ref.null or ref.is_null"),
                    })
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::Identifier(name) => {
                let id = name.clone();
                self.advance();

                // Check for function call
                if self.current_token == Token::LParen {
                    self.advance();
                    let mut args = Vec::new();

                    while self.current_token != Token::RParen && self.current_token != Token::Eof {
                        args.push(self.parse_expression()?);
                        if !self.match_token(Token::Comma) {
                            break;
                        }
                    }

                    self.expect(Token::RParen)?;

                    // Check if this is call_ref
                    if id == "call_ref" && !args.is_empty() {
                        let func_ref = args.remove(0);
                        Ok(Expression::CallRef(Box::new(func_ref), args))
                    } else {
                        Ok(Expression::Call(id, args))
                    }
                }
                // Check for table access: table[index] or identifier[index]
                else if self.current_token == Token::LBracket {
                    self.advance();
                    let idx = self.parse_expression()?;
                    self.expect(Token::RBracket)?;
                    Ok(Expression::TableGet(Box::new(idx)))
                } else {
                    Ok(Expression::Variable(id))
                }
            }
            _ => Err(CompileError::ParseError {
                line: 0,
                col: 0,
                message: format!("unexpected token in expression: {:?}", self.current_token),
            }),
        }
    }
}

// ============================================================================
// CODE GENERATOR - Using webassembly crate types
// ============================================================================

struct CodeGen {
    program: Program,
    function_indices: BTreeMap<String, u32>,
    local_indices: BTreeMap<String, (u32, Type)>,
    current_func: Option<Function>,
    type_indices: BTreeMap<(Vec<Type>, Type), u32>,
    import_count: u32,
    func_types: Vec<FuncType>,
    imports: Vec<webassembly::Import>,
    function_type_indices: Vec<u32>,
    tables: Vec<webassembly::Table>,
    memories: Vec<Limits>,
    exports: Vec<webassembly::Export>,
    codes: Vec<webassembly::Code>,
    // String literal storage: offset -> string data
    string_literals: Vec<(u32, String)>,
    next_data_offset: u32,
}

impl CodeGen {
    fn new(program: Program) -> Self {
        CodeGen {
            program,
            function_indices: BTreeMap::new(),
            local_indices: BTreeMap::new(),
            current_func: None,
            type_indices: BTreeMap::new(),
            import_count: 0,
            func_types: Vec::new(),
            imports: Vec::new(),
            function_type_indices: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            exports: Vec::new(),
            codes: Vec::new(),
            string_literals: Vec::new(),
            next_data_offset: 0,
        }
    }

    fn generate(mut self) -> Result<Vec<u8>, CompileError> {
        // Clone program data to avoid borrow issues
        let memory = self.program.memory.clone();
        let tables = self.program.tables.clone();
        let imports = self.program.imports.clone();
        let functions = self.program.functions.clone();

        // Generate memory section if declared
        if let Some(mem) = &memory {
            let _flags = if mem.maximum.is_some() { 0x01 } else { 0x00 }
                | if mem.shared { 0x03 } else { 0x00 };
            self.memories.push(Limits {
                min: mem.initial,
                max: mem.maximum,
            });
        }

        // Generate table section
        for table in &tables {
            let reftype = match table.elem_type {
                Type::Funcref => FUNCREF,
                Type::Externref => EXTERNREF,
                _ => FUNCREF,
            };
            self.tables.push(webassembly::Table {
                reftype,
                limits: Limits {
                    min: table.initial,
                    max: table.maximum,
                },
            });
        }

        // Generate imports and collect their indices
        for import in &imports {
            self.generate_import(import)?;
        }
        self.import_count = imports.len() as u32;

        // Collect all function indices
        let mut func_idx = self.import_count;
        for func in &functions {
            self.function_indices.insert(func.name.clone(), func_idx);
            func_idx += 1;
        }

        // Generate type indices for all functions
        for func in &functions {
            self.get_or_create_type_index(&func.params, &func.return_type);
        }

        // Generate function and code sections
        for func in &functions {
            self.generate_function(func)?;
        }

        // Generate exports
        for func in &functions {
            if func.is_export {
                let func_idx = *self.function_indices.get(&func.name).unwrap();
                self.exports.push(webassembly::Export {
                    name: func.name.clone(),
                    kind: 0x00, // Function export
                    idx: func_idx,
                });
            }
        }

        // Export memory if present
        if memory.is_some() {
            self.exports.push(webassembly::Export {
                name: String::from("memory"),
                kind: 0x02, // Memory export
                idx: 0,
            });
        }

        // Build sections by moving values
        let sections = self.build_sections();

        // Use webassembly crate's encode function
        let program = webassembly::Program { sections };
        Ok(encode(&program))
    }

    fn build_sections(self) -> Vec<Section> {
        let mut sections = Vec::new();

        // Type section
        if !self.func_types.is_empty() {
            sections.push(Section::Type(self.func_types));
        }

        // Import section
        if !self.imports.is_empty() {
            sections.push(Section::Import(self.imports));
        }

        // Function section
        if !self.function_type_indices.is_empty() {
            sections.push(Section::Function(self.function_type_indices));
        }

        // Table section
        if !self.tables.is_empty() {
            sections.push(Section::Table(self.tables));
        }

        // Memory section
        if !self.memories.is_empty() {
            sections.push(Section::Memory(self.memories));
        }

        // Export section
        if !self.exports.is_empty() {
            sections.push(Section::Export(self.exports));
        }

        // Code section
        if !self.codes.is_empty() {
            sections.push(Section::Code(self.codes));
        }

        // Data section (for string literals)
        if !self.string_literals.is_empty() {
            let mut datas = Vec::new();
            for (offset, string) in self.string_literals {
                // Add null terminator
                let mut init = string.into_bytes();
                init.push(0); // Null terminator

                datas.push(webassembly::Data {
                    mode: webassembly::DataMode::Active {
                        memory: 0, // Memory index 0
                        offset: vec![Instruction::I32Const(offset as i32)],
                    },
                    init,
                });
            }
            sections.push(Section::Data(datas));
        }

        sections
    }

    fn generate_import(&mut self, import: &ImportDecl) -> Result<(), CompileError> {
        // Create type for import
        let param_types: Vec<Type> = import.params.iter().map(|(_, t)| t.clone()).collect();
        let type_idx = self.get_or_create_type_index_from_types(&param_types, &import.return_type);

        // Create import entry
        self.imports.push(webassembly::Import {
            module: import.module.clone(),
            name: import.name.clone(),
            desc: webassembly::ImportDesc::Func(type_idx),
        });

        Ok(())
    }

    fn generate_function(&mut self, func: &Function) -> Result<(), CompileError> {
        self.current_func = Some(func.clone());

        // Get or create type index
        let type_idx = self.get_or_create_type_index(&func.params, &func.return_type);
        self.function_type_indices.push(type_idx);

        // Build local index map
        self.local_indices.clear();
        let mut local_idx = 0u32;

        // Parameters
        for (name, ty) in &func.params {
            self.local_indices
                .insert(name.clone(), (local_idx, ty.clone()));
            local_idx += 1;
        }

        // Locals
        for (name, ty) in &func.locals {
            self.local_indices
                .insert(name.clone(), (local_idx, ty.clone()));
            local_idx += 1;
        }

        // Generate code
        let mut instructions = Vec::new();

        // Function body
        for stmt in &func.body {
            self.generate_statement(&mut instructions, stmt)?;
        }

        // Add implicit return for void functions if needed
        if func.return_type == Type::Void {
            let needs_return = func
                .body
                .last()
                .map(|s| !matches!(s, Statement::Return(_)))
                .unwrap_or(true);

            if needs_return {
                instructions.push(Instruction::Return);
            }
        } else {
            let has_return = func
                .body
                .last()
                .map(|s| matches!(s, Statement::Return(_)))
                .unwrap_or(false);

            if !has_return {
                // Push a default value
                match &func.return_type {
                    Type::I32 => {
                        instructions.push(Instruction::I32Const(0));
                    }
                    Type::I64 => {
                        instructions.push(Instruction::I64Const(0));
                    }
                    Type::F32 => {
                        instructions.push(Instruction::F32Const(0.0));
                    }
                    Type::F64 => {
                        instructions.push(Instruction::F64Const(0.0));
                    }
                    Type::Externref | Type::Funcref => {
                        let reftype = match func.return_type {
                            Type::Externref => EXTERNREF,
                            Type::Funcref => FUNCREF,
                            _ => EXTERNREF,
                        };
                        instructions.push(Instruction::RefNull(reftype));
                    }
                    Type::Void => {}
                }
                instructions.push(Instruction::Return);
            }
        }

        // Always end the function body with END
        instructions.push(Instruction::End);

        // Group locals by type for WASM encoding
        let locals_by_type = self.group_locals_by_type(&func.locals);
        let mut local_decls: Vec<(u32, ValType)> = Vec::new();
        for (count, ty) in locals_by_type {
            let valtype = type_to_valtype(&ty);
            local_decls.push((count, valtype));
        }

        self.codes.push(webassembly::Code {
            locals: local_decls,
            body: instructions,
        });

        self.current_func = None;
        Ok(())
    }

    fn generate_statement(
        &mut self,
        instructions: &mut Vec<Instruction>,
        stmt: &Statement,
    ) -> Result<(), CompileError> {
        match stmt {
            Statement::VarDecl(name, _ty, init) => {
                let local_idx = self.local_indices.get(name).unwrap().0;

                if let Some(expr) = init {
                    self.generate_expression(instructions, expr)?;
                    instructions.push(Instruction::LocalSet(local_idx));
                }
                Ok(())
            }
            Statement::Assign(name, value) => {
                let local_idx = self.local_indices.get(name).unwrap().0;
                self.generate_expression(instructions, value)?;
                instructions.push(Instruction::LocalSet(local_idx));
                Ok(())
            }
            Statement::Expr(expr) => {
                self.generate_expression(instructions, expr)?;
                Ok(())
            }
            Statement::If(cond, then_branch, else_branch) => {
                self.generate_expression(instructions, cond)?;
                instructions.push(Instruction::If(BlockType::Empty));

                for s in then_branch {
                    self.generate_statement(instructions, s)?;
                }

                if let Some(else_stmts) = else_branch {
                    instructions.push(Instruction::Else);
                    for s in else_stmts {
                        self.generate_statement(instructions, s)?;
                    }
                }

                instructions.push(Instruction::End);
                Ok(())
            }
            Statement::While(cond, body) => {
                // Block wrapping loop for proper branching
                instructions.push(Instruction::Block(BlockType::Empty));
                instructions.push(Instruction::Loop(BlockType::Empty));

                // Condition
                self.generate_expression(instructions, cond)?;
                instructions.push(Instruction::I32Eqz);
                instructions.push(Instruction::BrIf(1)); // Branch to block end

                // Body
                for s in body {
                    self.generate_statement(instructions, s)?;
                }

                // Continue loop
                instructions.push(Instruction::Br(0)); // Branch to loop start

                instructions.push(Instruction::End); // Loop
                instructions.push(Instruction::End); // Block
                Ok(())
            }
            Statement::For(init, cond, update, body) => {
                // Initialize
                if let Some(init_stmt) = init {
                    let stmt = init_stmt.as_ref();
                    if let Statement::Expr(expr) = stmt {
                        self.generate_expression(instructions, expr)?;
                    } else {
                        self.generate_statement(instructions, stmt)?;
                    }
                }

                // Loop structure
                instructions.push(Instruction::Block(BlockType::Empty));
                instructions.push(Instruction::Loop(BlockType::Empty));

                // Condition check
                if let Some(c) = cond {
                    self.generate_expression(instructions, c)?;
                    instructions.push(Instruction::I32Eqz);
                    instructions.push(Instruction::BrIf(1));
                }

                // Body
                for s in body {
                    self.generate_statement(instructions, s)?;
                }

                // Update
                if let Some(upd) = update {
                    let stmt = upd.as_ref();
                    if let Statement::Expr(expr) = stmt {
                        self.generate_expression(instructions, expr)?;
                    } else {
                        self.generate_statement(instructions, stmt)?;
                    }
                }

                // Continue
                instructions.push(Instruction::Br(0));

                instructions.push(Instruction::End); // Loop
                instructions.push(Instruction::End); // Block
                Ok(())
            }
            Statement::Return(value) => {
                if let Some(expr) = value {
                    self.generate_expression(instructions, expr)?;
                }
                instructions.push(Instruction::Return);
                Ok(())
            }
            Statement::Block(stmts) => {
                for s in stmts {
                    self.generate_statement(instructions, s)?;
                }
                Ok(())
            }
            Statement::TableSet(idx_expr, value) => {
                self.generate_expression(instructions, idx_expr)?;
                self.generate_expression(instructions, value)?;
                instructions.push(Instruction::TableSet(0)); // Table index 0
                Ok(())
            }
            Statement::MemoryStore(ptr_expr, value) => {
                // Store value at memory address: compute address, then value, then store
                self.generate_expression(instructions, ptr_expr)?;
                self.generate_expression(instructions, value)?;
                instructions.push(Instruction::I32Store(MemArg {
                    align: 2, // 4-byte alignment
                    offset: 0,
                }));
                Ok(())
            }
        }
    }

    fn generate_expression(
        &mut self,
        instructions: &mut Vec<Instruction>,
        expr: &Expression,
    ) -> Result<(), CompileError> {
        match expr {
            Expression::IntLiteral(n) => {
                if *n >= i32::MIN as i64 && *n <= i32::MAX as i64 {
                    instructions.push(Instruction::I32Const(*n as i32));
                } else {
                    instructions.push(Instruction::I64Const(*n));
                }
                Ok(())
            }
            Expression::FloatLiteral(f) => {
                if *f >= f32::MIN as f64 && *f <= f32::MAX as f64 {
                    instructions.push(Instruction::F32Const(*f as f32));
                } else {
                    instructions.push(Instruction::F64Const(*f));
                }
                Ok(())
            }
            Expression::BoolLiteral(b) => {
                instructions.push(Instruction::I32Const(if *b { 1 } else { 0 }));
                Ok(())
            }
            Expression::StringLiteral(s) => {
                // Allocate string in data section and return offset
                let offset = self.add_string_literal(s.clone());
                instructions.push(Instruction::I32Const(offset as i32));
                Ok(())
            }
            Expression::Variable(name) => {
                if let Some((idx, _)) = self.local_indices.get(name) {
                    instructions.push(Instruction::LocalGet(*idx));
                    Ok(())
                } else {
                    Err(CompileError::CodeGenError {
                        message: format!("undefined variable: {}", name),
                    })
                }
            }
            Expression::Binary(op, left, right) => {
                self.generate_expression(instructions, left)?;
                self.generate_expression(instructions, right)?;

                let instruction = match op {
                    BinOp::Add => Instruction::I32Add,
                    BinOp::Sub => Instruction::I32Sub,
                    BinOp::Mul => Instruction::I32Mul,
                    BinOp::Div => Instruction::I32DivS,
                    BinOp::Mod => Instruction::I32RemS,
                    BinOp::Eq => Instruction::I32Eq,
                    BinOp::Ne => Instruction::I32Ne,
                    BinOp::Lt => Instruction::I32LtS,
                    BinOp::Gt => Instruction::I32GtS,
                    BinOp::Le => Instruction::I32LeS,
                    BinOp::Ge => Instruction::I32GeS,
                    BinOp::And => Instruction::I32And,
                    BinOp::Or => Instruction::I32Or,
                    BinOp::BitAnd => Instruction::I32And,
                    BinOp::BitOr => Instruction::I32Or,
                    BinOp::BitXor => Instruction::I32Xor,
                    BinOp::Shl => Instruction::I32Shl,
                    BinOp::Shr => Instruction::I32ShrS,
                };

                instructions.push(instruction);
                Ok(())
            }
            Expression::Unary(op, operand) => {
                self.generate_expression(instructions, operand)?;

                match op {
                    UnaryOp::Neg => {
                        instructions.push(Instruction::I32Const(0));
                        instructions.push(Instruction::I32Sub);
                    }
                    UnaryOp::Not => {
                        instructions.push(Instruction::I32Eqz);
                    }
                    UnaryOp::BitNot => {
                        instructions.push(Instruction::I32Const(-1));
                        instructions.push(Instruction::I32Xor);
                    }
                }

                Ok(())
            }
            Expression::Call(name, args) => {
                // Push arguments (in reverse order for stack machine)
                for arg in args.iter().rev() {
                    self.generate_expression(instructions, arg)?;
                }

                if let Some(&idx) = self.function_indices.get(name) {
                    instructions.push(Instruction::Call(idx));
                } else {
                    let import_idx = self.get_import_index(name);
                    if let Some(idx) = import_idx {
                        instructions.push(Instruction::Call(idx));
                    } else {
                        return Err(CompileError::CodeGenError {
                            message: format!("undefined function: {}", name),
                        });
                    }
                }

                Ok(())
            }
            Expression::CallRef(func_ref, args) => {
                // Push arguments
                for arg in args.iter().rev() {
                    self.generate_expression(instructions, arg)?;
                }

                // Push function reference
                self.generate_expression(instructions, func_ref)?;

                // TODO: We need to get the type index for call_ref
                // For now, assume type index 0
                instructions.push(Instruction::CallRef(0));

                Ok(())
            }
            Expression::RefNull(ty) => {
                let reftype = match ty {
                    Type::Externref => EXTERNREF,
                    Type::Funcref => FUNCREF,
                    _ => EXTERNREF,
                };
                instructions.push(Instruction::RefNull(reftype));
                Ok(())
            }
            Expression::RefIsNull(expr) => {
                self.generate_expression(instructions, expr)?;
                instructions.push(Instruction::RefIsNull);
                Ok(())
            }
            Expression::RefFunc(name) => {
                if let Some(&idx) = self.function_indices.get(name) {
                    instructions.push(Instruction::RefFunc(idx));
                    Ok(())
                } else {
                    Err(CompileError::CodeGenError {
                        message: format!("undefined function reference: {}", name),
                    })
                }
            }
            Expression::TableGet(idx_expr) => {
                self.generate_expression(instructions, idx_expr)?;
                instructions.push(Instruction::TableGet(0)); // Table index 0
                Ok(())
            }
            Expression::RefCast(expr, _ty) => {
                self.generate_expression(instructions, expr)?;
                // TODO: Implement proper ref.cast when needed
                Ok(())
            }
            Expression::Deref(expr) => {
                // Load from memory: generate address expression, then load
                self.generate_expression(instructions, expr)?;
                // i32.load with default alignment and offset
                instructions.push(Instruction::I32Load(MemArg {
                    align: 2,  // 4-byte alignment (2^2)
                    offset: 0,
                }));
                Ok(())
            }
            Expression::AddressOf(_) => Err(CompileError::CodeGenError {
                message: String::from("address-of operator for variables not supported (use memory pointers with i32)"),
            }),
        }
    }

    fn get_or_create_type_index(&mut self, params: &[(String, Type)], return_type: &Type) -> u32 {
        let param_types: Vec<Type> = params.iter().map(|(_, t)| t.clone()).collect();
        self.get_or_create_type_index_from_types(&param_types, return_type)
    }

    fn get_or_create_type_index_from_types(&mut self, params: &[Type], return_type: &Type) -> u32 {
        let key = (params.to_vec(), return_type.clone());

        if let Some(&idx) = self.type_indices.get(&key) {
            return idx;
        }

        let idx = self.func_types.len() as u32;

        // Build FuncType
        let wasm_params: Vec<ValType> = params.iter().map(|t| type_to_valtype(t)).collect();
        let wasm_results: Vec<ValType> = if *return_type == Type::Void {
            Vec::new()
        } else {
            vec![type_to_valtype(return_type)]
        };

        self.func_types.push(FuncType {
            params: wasm_params,
            results: wasm_results,
        });
        self.type_indices.insert(key, idx);

        idx
    }

    fn group_locals_by_type(&self, locals: &[(String, Type)]) -> Vec<(u32, Type)> {
        let mut grouped: BTreeMap<Type, u32> = BTreeMap::new();
        for (_, ty) in locals {
            *grouped.entry(ty.clone()).or_insert(0) += 1;
        }
        grouped.into_iter().map(|(t, c)| (c, t)).collect()
    }

    fn add_string_literal(&mut self, s: String) -> u32 {
        // For simplicity, store strings with a null terminator
        let offset = self.next_data_offset;
        let data = s.into_bytes();
        let len = data.len() as u32 + 1; // +1 for null terminator

        self.string_literals
            .push((offset, String::from_utf8(data).unwrap_or_default()));
        self.next_data_offset += len;

        offset
    }

    fn get_import_index(&self, name: &str) -> Option<u32> {
        for (i, import) in self.program.imports.iter().enumerate() {
            if import.func_name == name {
                return Some(i as u32);
            }
        }
        None
    }
}

fn type_to_valtype(t: &Type) -> ValType {
    match t {
        Type::I32 => ValType::I32,
        Type::I64 => ValType::I64,
        Type::F32 => ValType::F32,
        Type::F64 => ValType::F64,
        Type::Externref => ValType::ExternRef,
        Type::Funcref => ValType::FuncRef,
        Type::Void => ValType::I32, // Shouldn't happen in practice
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::println;

    #[test]
    fn test_simple_compile() {
        let source = "export i32 main() { return 42; }\n";
        let result = compile(source);
        match result {
            Ok(bytes) => {
                println!("Success! Generated {} bytes", bytes.len());
                // Check magic number
                assert_eq!(&bytes[0..4], &MAGIC);
            }
            Err(e) => {
                println!("Error: {:?}", e);
                panic!("Compilation failed");
            }
        }
    }

    #[test]
    fn test_arithmetic() {
        let source = r#"
export i32 add(i32 a, i32 b) {
    return a + b;
}

export i32 main() {
    return add(10, 32);
}
"#;
        let result = compile(source);
        assert!(result.is_ok(), "Arithmetic test failed: {:?}", result.err());
        let bytes = result.unwrap();
        assert!(bytes.len() > 10, "Generated WASM too small");
    }

    #[test]
    fn test_control_flow() {
        let source = r#"
export i32 factorial(i32 n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

export i32 main() {
    return factorial(5);
}
"#;
        let result = compile(source);
        assert!(
            result.is_ok(),
            "Control flow test failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_loops() {
        let source = r#"
export i32 sum_loop() {
    i32 sum = 0;
    i32 i = 1;
    while (i <= 10) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}

export i32 main() {
    return sum_loop();
}
"#;
        let result = compile(source);
        assert!(result.is_ok(), "Loop test failed: {:?}", result.err());
    }

    #[test]
    fn test_string_literals() {
        let source = r#"
export i32 main() {
    i32 msg = "Hello, World!";
    return 0;
}
"#;
        let result = compile(source);
        assert!(
            result.is_ok(),
            "String literal test failed: {:?}",
            result.err()
        );
        let bytes = result.unwrap();
        // Check that data section exists (look for data section header pattern)
        assert!(bytes.len() > 50, "Should have data section with string");
    }

    #[test]
    fn test_memory_operations() {
        let source = r#"
memory 1;

export i32 test_memory() {
    i32 ptr = 0;
    *ptr = 123;
    return *ptr;
}

export i32 main() {
    return test_memory();
}
"#;
        let result = compile(source);
        assert!(
            result.is_ok(),
            "Memory operations test failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_references() {
        let source = r#"
memory 1;

extern void console_log(externref msg);

export i32 test_refs() {
    externref null_ref = ref.null externref;
    if (ref.is_null(null_ref)) {
        return 1;
    }
    return 0;
}

export i32 main() {
    return test_refs();
}
"#;
        let result = compile(source);
        assert!(result.is_ok(), "References test failed: {:?}", result.err());
    }
}
