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

#![no_std]
extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use webassembly::op::*;
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
    let mut codegen = CodeGen::new(program);
    codegen.generate()
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

                // Assignment: var = expr;
                if self.current_token == Token::Assign {
                    if let Expression::Variable(name) = expr {
                        self.advance();
                        let value = self.parse_expression()?;
                        self.match_token(Token::Semicolon);
                        return Ok(Statement::Assign(name, value));
                    } else {
                        return Err(CompileError::ParseError {
                            line: 0,
                            col: 0,
                            message: String::from("invalid assignment target"),
                        });
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
                    Ok(Expression::TableGet(Box::new(Expression::Variable(id))))
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
// CODE GENERATOR
// ============================================================================

struct CodeGen {
    program: Program,
    module: WasmModule,
    function_indices: BTreeMap<String, u32>,
    local_indices: BTreeMap<String, (u32, Type)>,
    current_func: Option<Function>,
    type_indices: BTreeMap<(Vec<Type>, Type), u32>,
    import_count: u32,
}

struct WasmModule {
    types: Vec<Vec<u8>>,
    imports: Vec<Vec<u8>>,
    functions: Vec<u32>,
    tables: Vec<Vec<u8>>,
    memories: Vec<Vec<u8>>,
    exports: Vec<Vec<u8>>,
    codes: Vec<Vec<u8>>,
}

impl CodeGen {
    fn new(program: Program) -> Self {
        let mut module = WasmModule {
            types: Vec::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            exports: Vec::new(),
            codes: Vec::new(),
        };

        CodeGen {
            program,
            module,
            function_indices: BTreeMap::new(),
            local_indices: BTreeMap::new(),
            current_func: None,
            type_indices: BTreeMap::new(),
            import_count: 0,
        }
    }

    fn generate(&mut self) -> Result<Vec<u8>, CompileError> {
        // Clone program data to avoid borrow issues
        let memory = self.program.memory.clone();
        let tables = self.program.tables.clone();
        let imports = self.program.imports.clone();
        let functions = self.program.functions.clone();

        // Generate memory section if declared
        if let Some(mem) = &memory {
            let mut mem_bytes = Vec::new();
            let flags = if mem.maximum.is_some() { 0x01 } else { 0x00 }
                | if mem.shared { 0x03 } else { 0x00 };
            mem_bytes.push(flags);
            mem_bytes.extend(leb128_u32(mem.initial));
            if let Some(max) = mem.maximum {
                mem_bytes.extend(leb128_u32(max));
            }
            self.module.memories.push(mem_bytes);
        }

        // Generate table section
        for table in &tables {
            let mut table_bytes = Vec::new();
            let flags = if table.maximum.is_some() { 0x01 } else { 0x00 };
            table_bytes.push(flags);
            table_bytes.extend(leb128_u32(table.initial));
            if let Some(max) = table.maximum {
                table_bytes.extend(leb128_u32(max));
            }
            self.module.tables.push(table_bytes);
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
                let func_idx = self.function_indices.get(&func.name).unwrap();
                let mut export = Vec::new();
                export.push(func.name.len() as u8);
                export.extend_from_slice(func.name.as_bytes());
                export.push(0x00); // Export kind: function
                export.extend(leb128_u32(*func_idx));
                self.module.exports.push(export);
            }
        }

        // Export memory if present
        if memory.is_some() {
            let mut export = Vec::new();
            export.push(6); // "memory"
            export.extend_from_slice(b"memory");
            export.push(0x02); // Export kind: memory
            export.extend(leb128_u32(0)); // Memory index 0
            self.module.exports.push(export);
        }

        // Build final WASM binary
        let mut output = Vec::new();

        // Magic number and version
        output.extend_from_slice(&MAGIC);
        output.extend_from_slice(&VERSION.to_le_bytes());

        // Type section
        if !self.module.types.is_empty() {
            output.push(TYPE);
            let mut section_content = Vec::new();
            section_content.extend((self.module.types.len() as u32).to_wasm_bytes());
            for t in &self.module.types {
                section_content.extend(t);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Import section
        if !self.module.imports.is_empty() {
            output.push(IMPORT);
            let mut section_content = Vec::new();
            section_content.extend((self.module.imports.len() as u32).to_wasm_bytes());
            for i in &self.module.imports {
                section_content.extend(i);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Function section
        if !self.module.functions.is_empty() {
            output.push(FUNCTION);
            let mut section_content = Vec::new();
            section_content.extend((self.module.functions.len() as u32).to_wasm_bytes());
            for f in &self.module.functions {
                section_content.extend((*f as u32).to_wasm_bytes());
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Table section
        if !self.module.tables.is_empty() {
            output.push(TABLE);
            let mut section_content = Vec::new();
            section_content.extend((self.module.tables.len() as u32).to_wasm_bytes());
            for t in &self.module.tables {
                section_content.extend(t);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Memory section
        if !self.module.memories.is_empty() {
            output.push(MEMORY);
            let mut section_content = Vec::new();
            section_content.extend((self.module.memories.len() as u32).to_wasm_bytes());
            for m in &self.module.memories {
                section_content.extend(m);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Export section
        if !self.module.exports.is_empty() {
            output.push(EXPORT);
            let mut section_content = Vec::new();
            section_content.extend((self.module.exports.len() as u32).to_wasm_bytes());
            for e in &self.module.exports {
                section_content.extend(e);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        // Code section
        if !self.module.codes.is_empty() {
            output.push(CODE);
            let mut section_content = Vec::new();
            section_content.extend((self.module.codes.len() as u32).to_wasm_bytes());
            for c in &self.module.codes {
                section_content.extend(c);
            }
            output.extend((section_content.len() as u32).to_wasm_bytes());
            output.extend(section_content);
        }

        Ok(output)
    }

    fn generate_import(&mut self, import: &ImportDecl) -> Result<(), CompileError> {
        // Create type for import
        let param_types: Vec<Type> = import.params.iter().map(|(_, t)| t.clone()).collect();
        let type_idx = self.get_or_create_type_index_from_types(&param_types, &import.return_type);

        // Generate import entry
        let mut import_bytes = Vec::new();

        // Module name
        import_bytes.push(import.module.len() as u8);
        import_bytes.extend_from_slice(import.module.as_bytes());

        // Field name (function name)
        import_bytes.push(import.name.len() as u8);
        import_bytes.extend_from_slice(import.name.as_bytes());

        // Import kind (0 = function)
        import_bytes.push(0x00);

        // Type index
        import_bytes.extend(leb128_u32(type_idx));

        self.module.imports.push(import_bytes);

        Ok(())
    }

    fn generate_function(&mut self, func: &Function) -> Result<(), CompileError> {
        self.current_func = Some(func.clone());

        // Get or create type index
        let type_idx = self.get_or_create_type_index(&func.params, &func.return_type);
        self.module.functions.push(type_idx);

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
        let mut code = Vec::new();

        // Local declarations (count of locals by type)
        let locals_by_type = self.group_locals_by_type(&func.locals);
        code.extend(leb128_u32(locals_by_type.len() as u32));
        for (count, ty) in locals_by_type {
            code.extend(leb128_u32(count));
            code.push(type_to_valtype(&ty));
        }

        // Function body
        for stmt in &func.body {
            self.generate_statement(&mut code, stmt)?;
        }

        // Add implicit return for void functions if needed
        if func.return_type == Type::Void {
            // Check if last statement is not already a return
            let needs_return = func
                .body
                .last()
                .map(|s| !matches!(s, Statement::Return(_)))
                .unwrap_or(true);

            if needs_return {
                code.push(END);
            }
        } else {
            // Ensure we have a return
            let has_return = func
                .body
                .last()
                .map(|s| matches!(s, Statement::Return(_)))
                .unwrap_or(false);

            if !has_return {
                // Push a default value
                match &func.return_type {
                    Type::I32 => {
                        code.push(I32_CONST);
                        code.extend(leb128_i32(0));
                    }
                    Type::I64 => {
                        code.push(I64_CONST);
                        code.extend(leb128_i64(0));
                    }
                    Type::F32 => {
                        code.push(F32_CONST);
                        code.extend(&0.0f32.to_le_bytes());
                    }
                    Type::F64 => {
                        code.push(F64_CONST);
                        code.extend(&0.0f64.to_le_bytes());
                    }
                    Type::Externref | Type::Funcref => {
                        code.push(REF_NULL);
                        code.push(type_to_valtype(&func.return_type));
                    }
                    Type::Void => {}
                }
                code.push(END);
            }
        }

        // Wrap code in size
        let mut code_entry = Vec::new();
        code_entry.extend(leb128_u32(code.len() as u32));
        code_entry.extend(code);

        self.module.codes.push(code_entry);
        self.current_func = None;

        Ok(())
    }

    fn generate_statement(&self, code: &mut Vec<u8>, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::VarDecl(name, ty, init) => {
                let local_idx = self.local_indices.get(name).unwrap().0;

                if let Some(expr) = init {
                    self.generate_expression(code, expr)?;
                    code.push(LOCAL_SET);
                    code.extend(leb128_u32(local_idx));
                }
                Ok(())
            }
            Statement::Assign(name, value) => {
                let local_idx = self.local_indices.get(name).unwrap().0;
                self.generate_expression(code, value)?;
                code.push(LOCAL_SET);
                code.extend(leb128_u32(local_idx));
                Ok(())
            }
            Statement::Expr(expr) => {
                self.generate_expression(code, expr)?;
                // Drop result if not void
                if !matches!(expr, Expression::Call(_, _) | Expression::CallRef(_, _)) {
                    // Actually we need to check if the expression produces a value
                    // For now, just drop if it's not a call (calls might return void)
                }
                Ok(())
            }
            Statement::If(cond, then_branch, else_branch) => {
                self.generate_expression(code, cond)?;
                code.push(IF);
                code.push(0x40); // Empty block type (void)

                for s in then_branch {
                    self.generate_statement(code, s)?;
                }

                if let Some(else_stmts) = else_branch {
                    code.push(ELSE);
                    for s in else_stmts {
                        self.generate_statement(code, s)?;
                    }
                }

                code.push(END);
                Ok(())
            }
            Statement::While(cond, body) => {
                // Block wrapping loop for proper branching
                code.push(BLOCK);
                code.push(0x40); // Void

                code.push(LOOP);
                code.push(0x40); // Void

                // Condition
                self.generate_expression(code, cond)?;
                code.push(I32_EQZ); // Invert condition
                code.push(BR_IF);
                code.extend(leb128_u32(1)); // Branch to block end

                // Body
                for s in body {
                    self.generate_statement(code, s)?;
                }

                // Continue loop
                code.push(BR);
                code.extend(leb128_u32(0)); // Branch to loop start

                code.push(END); // Loop
                code.push(END); // Block
                Ok(())
            }
            Statement::For(init, cond, update, body) => {
                // Initialize
                if let Some(init_stmt) = init {
                    // This is already a boxed statement
                    let stmt = init_stmt.as_ref();
                    if let Statement::Expr(expr) = stmt {
                        self.generate_expression(code, expr)?;
                    } else {
                        self.generate_statement(code, stmt)?;
                    }
                }

                // Loop structure
                code.push(BLOCK);
                code.push(0x40); // Void

                code.push(LOOP);
                code.push(0x40); // Void

                // Condition check
                if let Some(c) = cond {
                    self.generate_expression(code, c)?;
                    code.push(I32_EQZ);
                    code.push(BR_IF);
                    code.extend(leb128_u32(1));
                }

                // Body
                for s in body {
                    self.generate_statement(code, s)?;
                }

                // Update
                if let Some(upd) = update {
                    let stmt = upd.as_ref();
                    if let Statement::Expr(expr) = stmt {
                        self.generate_expression(code, expr)?;
                    } else {
                        self.generate_statement(code, stmt)?;
                    }
                }

                // Continue
                code.push(BR);
                code.extend(leb128_u32(0));

                code.push(END); // Loop
                code.push(END); // Block
                Ok(())
            }
            Statement::Return(value) => {
                if let Some(expr) = value {
                    self.generate_expression(code, expr)?;
                }
                code.push(END);
                Ok(())
            }
            Statement::Block(stmts) => {
                for s in stmts {
                    self.generate_statement(code, s)?;
                }
                Ok(())
            }
            Statement::TableSet(idx_expr, value) => {
                // Table set: table index is always 0 for now
                self.generate_expression(code, idx_expr)?;
                self.generate_expression(code, value)?;
                code.push(TABLE_SET);
                code.extend(leb128_u32(0)); // Table index 0
                Ok(())
            }
        }
    }

    fn generate_expression(
        &self,
        code: &mut Vec<u8>,
        expr: &Expression,
    ) -> Result<(), CompileError> {
        match expr {
            Expression::IntLiteral(n) => {
                if *n >= i32::MIN as i64 && *n <= i32::MAX as i64 {
                    code.push(I32_CONST);
                    code.extend(leb128_i32(*n as i32));
                } else {
                    code.push(I64_CONST);
                    code.extend(leb128_i64(*n));
                }
                Ok(())
            }
            Expression::FloatLiteral(f) => {
                if *f >= f32::MIN as f64 && *f <= f32::MAX as f64 {
                    code.push(F32_CONST);
                    code.extend(&(*f as f32).to_le_bytes());
                } else {
                    code.push(F64_CONST);
                    code.extend(&f.to_le_bytes());
                }
                Ok(())
            }
            Expression::BoolLiteral(b) => {
                code.push(I32_CONST);
                code.extend(leb128_i32(if *b { 1 } else { 0 }));
                Ok(())
            }
            Expression::StringLiteral(_) => Err(CompileError::CodeGenError {
                message: String::from("string literals not yet implemented"),
            }),
            Expression::Variable(name) => {
                if let Some((idx, _)) = self.local_indices.get(name) {
                    code.push(LOCAL_GET);
                    code.extend(leb128_u32(*idx));
                    Ok(())
                } else {
                    Err(CompileError::CodeGenError {
                        message: format!("undefined variable: {}", name),
                    })
                }
            }
            Expression::Binary(op, left, right) => {
                self.generate_expression(code, left)?;
                self.generate_expression(code, right)?;

                let opcode = match op {
                    BinOp::Add => I32_ADD,
                    BinOp::Sub => I32_SUB,
                    BinOp::Mul => I32_MUL,
                    BinOp::Div => I32_DIV_S,
                    BinOp::Mod => I32_REM_S,
                    BinOp::Eq => I32_EQ,
                    BinOp::Ne => I32_NE,
                    BinOp::Lt => I32_LT_S,
                    BinOp::Gt => I32_GT_S,
                    BinOp::Le => I32_LE_S,
                    BinOp::Ge => I32_GE_S,
                    BinOp::And => I32_AND,
                    BinOp::Or => I32_OR,
                    BinOp::BitAnd => I32_AND,
                    BinOp::BitOr => I32_OR,
                    BinOp::BitXor => I32_XOR,
                    BinOp::Shl => I32_SHL,
                    BinOp::Shr => I32_SHR_S,
                };

                code.push(opcode);
                Ok(())
            }
            Expression::Unary(op, operand) => {
                self.generate_expression(code, operand)?;

                match op {
                    UnaryOp::Neg => {
                        code.push(I32_CONST);
                        code.extend(leb128_i32(0));
                        code.push(I32_SUB);
                    }
                    UnaryOp::Not => {
                        code.push(I32_EQZ);
                    }
                    UnaryOp::BitNot => {
                        code.push(I32_CONST);
                        code.extend(leb128_i32(-1));
                        code.push(I32_XOR);
                    }
                }

                Ok(())
            }
            Expression::Call(name, args) => {
                // Push arguments (in reverse order for stack machine)
                for arg in args.iter().rev() {
                    self.generate_expression(code, arg)?;
                }

                if let Some(&idx) = self.function_indices.get(name) {
                    code.push(CALL);
                    code.extend(leb128_u32(idx));
                } else {
                    // Might be an import - check by looking up
                    let import_idx = self.get_import_index(name);
                    if let Some(idx) = import_idx {
                        code.push(CALL);
                        code.extend(leb128_u32(idx));
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
                    self.generate_expression(code, arg)?;
                }

                // Push function reference
                self.generate_expression(code, func_ref)?;

                // TODO: We need to get the type index for call_ref
                // For now, assume type index 0
                code.push(CALL_REF);
                code.extend(leb128_u32(0)); // Type index

                Ok(())
            }
            Expression::RefNull(ty) => {
                code.push(REF_NULL);
                code.push(type_to_valtype(ty));
                Ok(())
            }
            Expression::RefIsNull(expr) => {
                self.generate_expression(code, expr)?;
                code.push(REF_IS_NULL);
                Ok(())
            }
            Expression::RefFunc(name) => {
                if let Some(&idx) = self.function_indices.get(name) {
                    code.push(REF_FUNC);
                    code.extend(leb128_u32(idx));
                    Ok(())
                } else {
                    Err(CompileError::CodeGenError {
                        message: format!("undefined function reference: {}", name),
                    })
                }
            }
            Expression::TableGet(idx_expr) => {
                self.generate_expression(code, idx_expr)?;
                code.push(TABLE_GET);
                code.extend(leb128_u32(0)); // Table index 0
                Ok(())
            }
            Expression::RefCast(expr, ty) => {
                self.generate_expression(code, expr)?;
                // TODO: Implement proper ref.cast when needed
                Ok(())
            }
            Expression::Deref(_) => Err(CompileError::CodeGenError {
                message: String::from("pointer dereference not yet implemented"),
            }),
            Expression::AddressOf(_) => Err(CompileError::CodeGenError {
                message: String::from("address-of operator not yet implemented"),
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

        let idx = self.module.types.len() as u32;

        // Build type entry
        let mut type_bytes = Vec::new();
        type_bytes.push(0x60); // Function type
        type_bytes.extend(leb128_u32(params.len() as u32));
        for p in params {
            type_bytes.push(type_to_valtype(p));
        }

        let result_count = if *return_type == Type::Void { 0 } else { 1 };
        type_bytes.extend(leb128_u32(result_count));
        if result_count > 0 {
            type_bytes.push(type_to_valtype(return_type));
        }

        self.module.types.push(type_bytes);
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

    fn get_import_index(&self, name: &str) -> Option<u32> {
        // Find import by function name
        for (i, import) in self.program.imports.iter().enumerate() {
            if import.func_name == name {
                return Some(i as u32);
            }
        }
        None
    }
}

fn type_to_valtype(t: &Type) -> u8 {
    match t {
        Type::I32 => I32,
        Type::I64 => I64,
        Type::F32 => F32,
        Type::F64 => F64,
        Type::Externref => EXTERNREF,
        Type::Funcref => FUNCREF,
        Type::Void => 0x40, // Empty block type
    }
}

// LEB128 encoding helpers
fn leb128_u32(mut value: u32) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        result.push(byte);
        if value == 0 {
            break;
        }
    }
    result
}

fn leb128_i32(mut value: i32) -> Vec<u8> {
    let mut result = Vec::new();
    let mut more = true;
    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        result.push(byte);
    }
    result
}

fn leb128_i64(mut value: i64) -> Vec<u8> {
    let mut result = Vec::new();
    let mut more = true;
    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        result.push(byte);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_lexer_tokens() {
        let source = "export i32 main() { return 42; }\n";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        println!("Tokens: {:?}", tokens);
    }
}
