fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("usage: wasc <file.wasc>"); return; }
    let src = std::fs::read_to_string(&args[1]).unwrap();
    match wasc::wasc_compile(&src) {
        Ok(bytes) => {
            std::fs::write("out.wasm", bytes).unwrap();
            println!("✅ wrote {} bytes → out.wasm", bytes.len());
        }
        Err(e) => eprintln!("❌ {}", e),
    }
}