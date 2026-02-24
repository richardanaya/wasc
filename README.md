# wasc - WebAssembly Single-file Compiler

A C-like language compiler specifically designed for WebAssembly with first-class support for references, tables, and direct memory management.

## Features

- **C-like syntax** - Familiar to C programmers
- **First-class externref and funcref types** - Easy interop with JavaScript
- **Direct memory and table management** - Explicit control over WASM resources
- **Zero-overhead external function calls** - Import and call JS functions directly
- **No standard library** - You control everything, no hidden overhead
- **Single-file implementation** - The entire compiler is in one Rust file

## Usage

```bash
wasc <input.wasc> [options]

Options:
  -o, --output <file>  Output file (default: input.wasm)
  -v, --verbose        Verbose output
  -h, --help           Show this help
```

## Example

```wasc
// Declare memory
memory 1;

// Declare table for function references
table funcref 10;

// Import external JS function
extern i32 js_add(i32 a, i32 b);

// Simple function
export i32 add(i32 a, i32 b) {
    return a + b;
}

// Main entry point
export i32 main() {
    i32 x = 10;
    i32 y = 32;
    return add(x, y);
}
```

Compile:
```bash
wasc example.wasc -o example.wasm
```

## Language Features

### Types
- `void` - No return value
- `i32`, `i64` - Integer types
- `f32`, `f64` - Floating point types
- `externref` - External reference (JS objects, etc.)
- `funcref` - Function reference for indirect calls

### Memory
```wasc
memory 1;           // 1 page (64KB)
memory 1, 10;       // Min 1 page, max 10 pages
memory 1 shared;    // Shared memory for threads
```

### Tables
```wasc
table funcref 10;        // Table of function references
table funcref 10, 100;   // Min 10, max 100
table externref 64;      // Table of external references
```

### Imports
```wasc
extern externref console_log(externref msg);
extern i32 js_add(i32 a, i32 b);
extern void do_something();
```

### Exports
```wasc
export i32 main() {
    return 42;
}
```

### Variables and Assignment
```wasc
i32 x = 10;
i32 y;
y = 20;
externref obj = ref.null externref;
```

### Control Flow
```wasc
if (x > 0) {
    return x;
} else {
    return 0;
}

while (i < 10) {
    i = i + 1;
}

for (i32 i = 0; i < 10; i = i + 1) {
    // loop body
}

return 42;
```

### Reference Operations
```wasc
// Create null reference
externref null_ref = ref.null externref;

// Check if null
if (ref.is_null(null_ref)) {
    return 1;
}

// Get function reference
funcref my_func = &add;

// Table operations
table[0] = &my_callback;
funcref callback = table[0];
```

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`

## Building

```bash
cargo build --release
```

The binary will be at `target/release/wasc`.

## WebAssembly Output

The compiler generates valid WebAssembly binary format that can be run in browsers, Node.js, or other WASM runtimes.

Example JavaScript usage:
```javascript
const wasmModule = await WebAssembly.instantiateStreaming(
    fetch("example.wasm"),
    {
        env: {
            js_add: (a, b) => a + b,
            memory: new WebAssembly.Memory({ initial: 1 })
        }
    }
);

const result = wasmModule.instance.exports.main();
console.log(result); // 42
```

## Architecture

The compiler consists of three main parts:
1. **Lexer** - Tokenizes the source code into tokens
2. **Parser** - Builds an AST from the tokens
3. **Code Generator** - Emits WebAssembly binary code

All implemented in a single Rust file with `no_std` support for minimal dependencies.

## License

MIT License - See LICENSE file for details
