use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    // Parse arguments
    let mut input_file = None;
    let mut output_file = None;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: -o flag requires a filename");
                    return;
                }
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            "-h" | "--help" => {
                print_usage();
                return;
            }
            _ => {
                if input_file.is_none() && !args[i].starts_with("-") {
                    input_file = Some(args[i].clone());
                }
                i += 1;
            }
        }
    }

    let input_file = match input_file {
        Some(f) => f,
        None => {
            eprintln!("Error: No input file specified");
            print_usage();
            return;
        }
    };

    let output_file = output_file.unwrap_or_else(|| {
        // Default to input file with .wasm extension
        if input_file.ends_with(".wasc") {
            input_file[..input_file.len() - 5].to_string() + ".wasm"
        } else {
            input_file.clone() + ".wasm"
        }
    });

    if verbose {
        println!("wasc: Compiling {} -> {}", input_file, output_file);
    }

    // Read source file
    let src = match fs::read_to_string(&input_file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading '{}': {}", input_file, e);
            return;
        }
    };

    // Compile
    match wasc::compile(&src) {
        Ok(bytes) => {
            if verbose {
                println!("wasc: Generated {} bytes", bytes.len());
            }

            // Write output
            match fs::write(&output_file, bytes) {
                Ok(_) => {
                    println!("wasc: Successfully compiled to {}", output_file);
                }
                Err(e) => {
                    eprintln!("Error writing '{}': {}", output_file, e);
                }
            }
        }
        Err(e) => {
            eprintln!("wasc: Compilation failed");
            eprintln!("  {:?}", e);
        }
    }
}

fn print_usage() {
    println!("wasc - WebAssembly Single-file Compiler");
    println!();
    println!("Usage: wasc <input.wasc> [options]");
    println!();
    println!("Options:");
    println!("  -o, --output <file>  Output file (default: input.wasm)");
    println!("  -v, --verbose        Verbose output");
    println!("  -h, --help           Show this help");
    println!();
    println!("Example:");
    println!("  wasc hello.wasc -o hello.wasm");
}
