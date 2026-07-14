use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp::{standard_sampler::StandardSampler, LlamaModel, LlamaParams, SessionParams};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    models: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 160)]
    max_tokens: usize,
    #[arg(long, default_value_t = 99)]
    gpu_layers: u32,
}

#[derive(Deserialize)]
struct ModelsFile {
    gguf: Vec<GgufModel>,
}

#[derive(Deserialize)]
struct GgufModel {
    id: String,
    path_env: String,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    purpose: String,
    prompt: String,
}

#[derive(Serialize)]
struct Row {
    runtime: &'static str,
    model_id: String,
    prompt_id: String,
    purpose: String,
    model_path: String,
    load_seconds: f64,
    generation_seconds: f64,
    cpu_seconds: f64,
    peak_rss_mb: f64,
    tokens_generated: usize,
    tokens_per_second: f64,
    output: String,
}

fn usage() -> (f64, f64) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return (0.0, 0.0);
    }
    let usage = unsafe { usage.assume_init() };
    let user = Duration::new(usage.ru_utime.tv_sec as u64, (usage.ru_utime.tv_usec * 1000) as u32);
    let system = Duration::new(
        usage.ru_stime.tv_sec as u64,
        (usage.ru_stime.tv_usec * 1000) as u32,
    );
    let peak_rss_mb = if usage.ru_maxrss > 10_000_000 {
        usage.ru_maxrss as f64 / 1024.0 / 1024.0
    } else {
        usage.ru_maxrss as f64 / 1024.0
    };
    ((user + system).as_secs_f64(), peak_rss_mb)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let models: ModelsFile = serde_json::from_reader(File::open(&args.models)?)?;
    let prompts: Vec<Prompt> = serde_json::from_reader(File::open(&args.prompts)?)?;
    let mut writer = BufWriter::new(File::create(&args.output)?);

    for model_entry in models.gguf {
        let Ok(model_path) = env::var(&model_entry.path_env) else {
            eprintln!("Skipping {} because {} is not set", model_entry.id, model_entry.path_env);
            continue;
        };

        let mut params = LlamaParams::default();
        params.n_gpu_layers = args.gpu_layers;

        let load_start = Instant::now();
        let model = match LlamaModel::load_from_file(&model_path, params)
            .with_context(|| format!("loading GGUF model at {model_path}"))
        {
            Ok(model) => model,
            Err(error) => {
                eprintln!("Skipping {} after load failure: {error:#}", model_entry.id);
                continue;
            }
        };
        let load_seconds = load_start.elapsed().as_secs_f64();

        for prompt in &prompts {
            let mut session = model.create_session(SessionParams::default())?;
            session.advance_context(&prompt.prompt)?;

            let generation_start = Instant::now();
            let mut output = String::new();
            let mut tokens_generated = 0usize;
            let (cpu_before, _) = usage();
            let completions = session
                .start_completing_with(StandardSampler::default(), args.max_tokens)
                ?
                .into_strings();

            for token in completions {
                output.push_str(&token);
                tokens_generated += 1;
                if tokens_generated >= args.max_tokens {
                    break;
                }
            }

            let generation_seconds = generation_start.elapsed().as_secs_f64();
            let (cpu_after, peak_rss_mb) = usage();
            let row = Row {
                runtime: "rust_llama_cpp",
                model_id: model_entry.id.clone(),
                prompt_id: prompt.id.clone(),
                purpose: prompt.purpose.clone(),
                model_path: model_path.clone(),
                load_seconds,
                generation_seconds,
                cpu_seconds: cpu_after - cpu_before,
                peak_rss_mb,
                tokens_generated,
                tokens_per_second: if generation_seconds > 0.0 {
                    tokens_generated as f64 / generation_seconds
                } else {
                    0.0
                },
                output,
            };
            serde_json::to_writer(&mut writer, &row)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
    }

    Ok(())
}
