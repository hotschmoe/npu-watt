use std::io::{BufRead, BufReader, Write};
use std::os::windows::io::AsRawHandle;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use npu_watt::*;

const USAGE: &str = "npu-watt: per-rail power for NPU-pinned inference (Windows Energy Meter)

  npu-watt monitor [--interval-ms N]
      live watts per rail

  npu-watt run [--label L] [--json F] [--csv F] [--tokens N] [--tok-s X] [--idle-secs S] [--interval-ms N] -- <cmd> [args]
      measure idle, run the command (whole process tree attributed for CPU), report NPU-active
      window, verdict, J/token, tok/s/W and battery. tokens and tok/s are parsed from llama.cpp's
      `eval time` line unless given.

  npu-watt record [--label L] [--idle-secs S] [--interval-ms N] [--pid P]...
      for scripts: measures idle, prints READY, records until stdin closes or a `stop` line.
      Stdin lines before stop: tokens=N, tok_s=X, label=L, eval=<llama.cpp eval line>.
      Prints the JSON report as the last stdout line (text report on stderr).

  npu-watt compare A.json B.json [--json OUT.json]
      B vs A: change in decode speed, power, J/token, tok/s/W";

struct Args {
    label: String,
    json: Option<String>,
    csv: Option<String>,
    tokens: Option<f64>,
    tok_s: Option<f64>,
    idle_secs: f64,
    interval: Duration,
    pids: Vec<u32>,
    cmd: Vec<String>,
    files: Vec<String>,
}

fn parse(a: &[String]) -> Result<Args, String> {
    let mut o = Args { label: "run".into(), json: None, csv: None, tokens: None, tok_s: None, idle_secs: 2.0, interval: Duration::from_millis(250), pids: vec![], cmd: vec![], files: vec![] };
    let mut i = 0;
    while i < a.len() {
        let val = |i: usize| a.get(i + 1).cloned().ok_or(format!("{} needs a value", a[i]));
        let num = |i: usize| -> Result<f64, String> { val(i)?.parse().map_err(|_| format!("bad number for {}", a[i])) };
        match a[i].as_str() {
            "--" => {
                o.cmd = a[i + 1..].to_vec();
                break;
            }
            "--label" => o.label = val(i)?,
            "--json" => o.json = Some(val(i)?),
            "--csv" => o.csv = Some(val(i)?),
            "--tokens" => o.tokens = Some(num(i)?),
            "--tok-s" => o.tok_s = Some(num(i)?),
            "--idle-secs" => o.idle_secs = num(i)?,
            "--interval-ms" => o.interval = Duration::from_millis(num(i)? as u64),
            "--pid" => o.pids.push(val(i)?.parse().map_err(|_| "bad --pid")?),
            x if !x.starts_with("--") => {
                o.files.push(x.to_string());
                i += 1;
                continue;
            }
            x => return Err(format!("unknown argument {x}")),
        }
        i += 2;
    }
    Ok(o)
}

fn monitor(interval: Duration) -> Result<(), String> {
    let m = Meter::open()?;
    println!("watts per rail (Energy Meter). Ctrl+C to stop.");
    println!("{:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>9}", "t(s)", "npu", "gpu", "cpu", "mem", "soc", "system", "batt-out");
    let t0 = Instant::now();
    loop {
        std::thread::sleep(interval);
        let (_, p) = m.sample()?;
        let g = |k: &str| p.get(k).copied().unwrap_or(0.0);
        let cpu: f64 = p.iter().filter(|(k, _)| k.starts_with("cpu_cluster")).map(|(_, v)| v).sum();
        let b = battery().map(|b| format!("{:+.1}", b.discharge_w)).unwrap_or_else(|| "-".into());
        println!("{:8.1} {:7.2} {:7.2} {:7.2} {:7.2} {:7.2} {:8.2} {:>9}", t0.elapsed().as_secs_f64(), g("npu"), g("gpu"), cpu, g("memory"), g("soc"), g("system"), b);
    }
}

fn write_csv(path: &str, samples: &[Sample]) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let rails: Vec<String> = samples[0].power_w.keys().cloned().collect();
    writeln!(f, "t_s,cpu_s,batt_w,{}", rails.iter().map(|r| format!("{r}_W")).collect::<Vec<_>>().join(",")).ok();
    for s in samples {
        let v: Vec<String> = rails.iter().map(|r| format!("{:.4}", s.power_w.get(r).copied().unwrap_or(0.0))).collect();
        writeln!(f, "{:.3},{:.3},{},{}", s.t, s.cpu_s, s.batt_w.map(|x| format!("{x:.3}")).unwrap_or_default(), v.join(",")).ok();
    }
    Ok(())
}

fn finish(o: &Args, samples: &[Sample], idle: &Rails, tokens: Option<f64>, tok_s: Option<f64>) -> Result<Report, String> {
    let r = analyze(samples, idle, &AnalyzeOpts { label: o.label.clone(), tokens, tok_s, ..Default::default() });
    if let Some(p) = &o.csv {
        write_csv(p, samples)?;
        eprintln!("[npu-watt] wrote {p}");
    }
    if let Some(p) = &o.json {
        std::fs::write(p, serde_json::to_string_pretty(&r).unwrap()).map_err(|e| e.to_string())?;
        eprintln!("[npu-watt] wrote {p}");
    }
    Ok(r)
}

fn run(o: Args) -> Result<(), String> {
    if o.cmd.is_empty() {
        return Err("run needs a command after --".into());
    }
    eprintln!("[npu-watt] measuring idle baseline for {:.1}s ...", o.idle_secs);
    let idle = measure_idle(Duration::from_secs_f64(o.idle_secs))?;

    let job = Arc::new(Job::new()?);
    let mut child = Command::new(&o.cmd[0]).args(&o.cmd[1..]).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("spawn {}: {e}", o.cmd[0]))?;
    // Descendants are picked up automatically; the child may briefly run before assignment.
    job.assign_raw(child.as_raw_handle())?;
    let rec = Recorder::start(o.interval, CpuSource::Job(job.clone()))?;

    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut readers = Vec::new();
    let streams: Vec<Box<dyn std::io::Read + Send>> = vec![Box::new(child.stdout.take().unwrap()), Box::new(child.stderr.take().unwrap())];
    for stream in streams {
        let lines = lines.clone();
        readers.push(std::thread::spawn(move || {
            let mut r = BufReader::new(stream);
            let mut buf = Vec::new();
            while r.read_until(b'\n', &mut buf).unwrap_or(0) > 0 {
                let l = String::from_utf8_lossy(&buf).trim_end().to_string();
                println!("{l}");
                lines.lock().unwrap().push(l);
                buf.clear();
            }
        }));
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    for r in readers {
        let _ = r.join();
    }
    let samples = rec.stop()?;
    eprintln!("[npu-watt] command exited: {status}");

    let parsed = parse_eval(&lines.lock().unwrap());
    let tokens = o.tokens.or(parsed.map(|p| p.0));
    let tok_s = o.tok_s.or(parsed.map(|p| p.1));
    let r = finish(&o, &samples, &idle, tokens, tok_s)?;
    println!("\n{}", r.render_text());
    if tok_s.is_none() {
        println!("(no token rate found; pass --tokens N --tok-s X to get J/token and tok/s/W)");
    }
    Ok(())
}

fn record(mut o: Args) -> Result<(), String> {
    let idle = measure_idle(Duration::from_secs_f64(o.idle_secs))?;
    let cpu = if o.pids.is_empty() { CpuSource::None } else { CpuSource::Pids(o.pids.clone()) };
    let rec = Recorder::start(o.interval, cpu)?;
    println!("READY");
    std::io::stdout().flush().ok();

    let mut eval_lines = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line == "stop" {
            break;
        } else if let Some(v) = line.strip_prefix("tokens=") {
            o.tokens = v.parse().ok();
        } else if let Some(v) = line.strip_prefix("tok_s=") {
            o.tok_s = v.parse().ok();
        } else if let Some(v) = line.strip_prefix("label=") {
            o.label = v.to_string();
        } else if let Some(v) = line.strip_prefix("eval=") {
            eval_lines.push(v.to_string());
        }
    }
    let samples = rec.stop()?;
    let parsed = parse_eval(&eval_lines);
    let tokens = o.tokens.or(parsed.map(|p| p.0));
    let tok_s = o.tok_s.or(parsed.map(|p| p.1));
    let r = finish(&o, &samples, &idle, tokens, tok_s)?;
    eprintln!("{}", r.render_text());
    println!("{}", serde_json::to_string(&r).unwrap());
    Ok(())
}

fn compare_cmd(o: Args) -> Result<(), String> {
    if o.files.len() != 2 {
        return Err("compare needs two report JSON files".into());
    }
    let load = |p: &str| -> Result<Report, String> { serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}")) };
    let (a, b) = (load(&o.files[0])?, load(&o.files[1])?);
    let c = compare(&a, &b);
    println!("{}", c.summary);
    if let Some(p) = &o.json {
        std::fs::write(p, serde_json::to_string_pretty(&c).unwrap()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(String::as_str) {
        Some(mode @ ("monitor" | "run" | "record" | "compare")) => parse(&a[1..]).and_then(|o| match mode {
            "monitor" => monitor(o.interval),
            "run" => run(o),
            "record" => record(o),
            _ => compare_cmd(o),
        }),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = r {
        eprintln!("npu-watt: {e}");
        std::process::exit(1);
    }
}
