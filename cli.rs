//! `sg` — the command-line front end over [`search::dispatch`].
//!
//! Three verbs: `info` describes a file, `scan` scores every node, `search`
//! walks the graph. Each takes a path or a glob, so a dataset sharded across
//! files is one argument.
//!
//! Everything the run needs comes from the file: the metric from
//! `__metadata__.metric`, the scalar from the value column's dtype, the axes
//! from its [`Shape`]. `scan` alone accepts `--metric` to override the
//! declaration — re-scoring one dataset under another metric is the point of a
//! brute-force pass — while `search` refuses it, because a graph's edges were
//! built for one metric and walking them under another quietly returns wrong
//! neighbors.
//!
//! Queries are JSON, inline (`--query`) or from a file (`--queries`); both share
//! one parser and differ only in where the bytes come from.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use safegraphs::search::dispatch::{is_ragged, ranking, ScannerAny, SearcherAny, METRICS};
use safegraphs::search::queries;
use safegraphs::search::{ExecutionScope, Matrix, Ranking};
use safegraphs::{Error, GraphLike, Shape, Table};

// region: Arguments

#[derive(Parser)]
#[command(name = "sg", about = "Search SafeGraphs files", version)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Describe one or more files: metric, dtype, shape, size.
    Info {
        /// A path, or a glob matching several shards of one dataset.
        index: String,
    },
    /// Brute-force top-k over every node.
    Scan {
        index: String,
        #[command(flatten)]
        query: QueryArguments,
        /// Score under this metric instead of the file's own. The value column's
        /// dtype and shape must suit it, or construction fails.
        #[arg(long)]
        metric: Option<String>,
        #[command(flatten)]
        run: RunArguments,
    },
    /// Top-k by graph walk.
    Search {
        index: String,
        #[command(flatten)]
        query: QueryArguments,
        /// Walk under a metric other than the file's. Requires `--force`: the
        /// edges were built for the declared metric.
        #[arg(long)]
        metric: Option<String>,
        /// Permit `--metric` on a walk, accepting wrong neighbors.
        #[arg(long)]
        force: bool,
        /// Frontier breadth. Larger widens the search and raises recall.
        #[arg(long, default_value_t = 64)]
        expansion: usize,
        #[command(flatten)]
        run: RunArguments,
    },
}

#[derive(clap::Args)]
struct QueryArguments {
    /// Queries as an inline JSON array.
    #[arg(long, conflicts_with = "queries")]
    query: Option<String>,
    /// Queries as a JSON file.
    #[arg(long)]
    queries: Option<PathBuf>,
}

#[derive(clap::Args)]
struct RunArguments {
    /// Neighbors to return per query.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Worker threads; 0 uses every logical core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Output shape: table, json, ndjson.
    #[arg(long, default_value = "table")]
    format: String,
}

// endregion: Arguments

// region: Query input

/// Read the query bytes a run was given, inline or from a file.
fn query_text(arguments: &QueryArguments) -> Result<Vec<u8>, String> {
    match (&arguments.query, &arguments.queries) {
        (Some(inline), _) => Ok(inline.clone().into_bytes()),
        (None, Some(path)) => {
            std::fs::read(path).map_err(|error| format!("reading {}: {error}", path.display()))
        }
        (None, None) => Err("supply queries with --query or --queries".to_string()),
    }
}

// endregion: Query input

// region: Shards

/// Expand a path or glob into the shards of one dataset, in a stable order.
fn expand(pattern: &str) -> Result<Vec<PathBuf>, String> {
    if !pattern.contains(['*', '?', '[']) {
        return Ok(vec![PathBuf::from(pattern)]);
    }
    let mut paths: Vec<PathBuf> = glob::glob(pattern)
        .map_err(|error| format!("glob({pattern}): {error}"))?
        .filter_map(Result::ok)
        .collect();
    if paths.is_empty() {
        return Err(format!("glob({pattern}) matched no files"));
    }
    paths.sort();
    Ok(paths)
}

/// Open every shard and check they describe one dataset. Differing dtypes or
/// shapes are a mismatch worth naming, since scoring across them is meaningless.
fn open_shards(paths: &[PathBuf]) -> Result<Vec<Table>, String> {
    let mut tables = Vec::new();
    for path in paths {
        let table = Table::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        if let Some(first) = tables.first() {
            let first: &Table = first;
            if first.value_dtype() != table.value_dtype() {
                return Err(format!(
                    "{}: dtype {} does not match the first shard's {}",
                    path.display(),
                    table.value_dtype(),
                    first.value_dtype()
                ));
            }
            if first.value_shape() != table.value_shape() {
                return Err(format!(
                    "{}: value shape does not match the first shard's",
                    path.display()
                ));
            }
        }
        tables.push(table);
    }
    Ok(tables)
}

// endregion: Shards

// region: Verbs

fn run_info(index: &str) -> Result<String, String> {
    let paths = expand(index)?;
    let tables = open_shards(&paths)?;
    let mut out = String::new();
    for (path, table) in paths.iter().zip(&tables) {
        let shape = match table.value_shape() {
            Some(Shape::Scalar) => "scalar".to_string(),
            Some(Shape::Vector(width)) => format!("[{width}]"),
            Some(Shape::Matrix(0, cols)) => format!("[*, {cols}]"),
            Some(Shape::Matrix(rows, cols)) => format!("[{rows}, {cols}]"),
            None => "none".to_string(),
        };
        out.push_str(&format!(
            "{}\n  metric    {}\n  dtype     {}\n  shape     {}\n  nodes     {}\n  degree    {}\n",
            path.display(),
            table.metric().unwrap_or("(undeclared)"),
            table.value_dtype(),
            shape,
            table.len(),
            table
                .neighbors_degree()
                .map_or("variable".to_string(), |degree| degree.to_string()),
        ));
    }
    Ok(out)
}

/// One shard's raw outputs: which shard, its `queries × k` matches and scores,
/// and how many each query actually found.
struct ShardResults {
    shard: usize,
    matches: Matrix<u64>,
    distances: Matrix<f64>,
    counts: Vec<u64>,
}

/// Merge each shard's ranked rows into one ranking per query, translating a
/// node into `(shard, node)` so a merged hit still names its file.
fn merge(
    per_shard: Vec<ShardResults>,
    tables: &[Table],
    queries: usize,
    k: usize,
    ranking: Ranking,
) -> Vec<Vec<Hit>> {
    let mut merged = Vec::with_capacity(queries);
    for query in 0..queries {
        let mut hits: Vec<Hit> = Vec::new();
        for shard_results in &per_shard {
            let ShardResults {
                shard,
                matches,
                distances,
                counts,
            } = shard_results;
            let found = counts[query] as usize;
            let (Some(nodes), Some(scores)) = (matches.row(query), distances.row(query)) else {
                continue;
            };
            for rank in 0..found.min(nodes.len()) {
                let node = nodes[rank];
                if node == u64::MAX {
                    continue;
                }
                hits.push(Hit {
                    shard: *shard,
                    node,
                    key: tables[*shard].key(node as usize).to_vec(),
                    score: scores[rank],
                });
            }
        }
        // `Ranking::compare` is a priority comparator — `Greater` means the left
        // score is the better match — so sorting by it directly would put the
        // worst first. Reverse it to get the best-first order `insert_at` builds.
        hits.sort_by(|a, b| ranking.compare(a.score, b.score).reverse());
        hits.truncate(k);
        merged.push(hits);
    }
    merged
}

#[allow(clippy::too_many_arguments)]
fn run_query(
    index: &str,
    query: &QueryArguments,
    metric_override: Option<&str>,
    run: &RunArguments,
    walk: bool,
    expansion: usize,
) -> Result<String, String> {
    let format = Format::parse(&run.format)
        .ok_or_else(|| format!("unknown --format {}; try table, json, ndjson", run.format))?;
    let paths = expand(index)?;
    let tables = open_shards(&paths)?;
    let first = tables.first().ok_or("no shards")?;

    let declared = first.metric();
    let metric = match metric_override {
        Some(name) => name,
        None => declared.ok_or_else(|| "the file declares no metric; pass --metric".to_string())?,
    };
    let dtype = first.value_dtype().to_string();
    let shape = first.value_shape().unwrap_or(Shape::Scalar);
    let direction = ranking(metric)
        .ok_or_else(|| format!("unknown metric {metric}; known: {}", METRICS.join(", ")))?;

    let parsed = queries::parse(&query_text(query)?, metric, &dtype, shape)
        .map_err(|error| error.to_string())?;
    if parsed.count() == 0 {
        return Err("no queries supplied".to_string());
    }
    if is_ragged(metric) && parsed.offsets().is_none() {
        return Err(format!("{metric} needs ragged queries"));
    }
    let scope = ExecutionScope::Cpus(run.threads);

    let mut per_shard = Vec::new();
    for (shard, table) in tables.iter().enumerate() {
        let mut matches =
            Matrix::<u64>::new(parsed.count(), run.k).map_err(|error| error.to_string())?;
        let mut distances =
            Matrix::<f64>::new(parsed.count(), run.k).map_err(|error| error.to_string())?;
        let mut counts = vec![0u64; parsed.count()];
        let offsets = parsed.offsets();
        if walk {
            let mut searcher =
                SearcherAny::new(metric, &dtype, expansion, scope, std::alloc::System)
                    .map_err(|error| error.to_string())?;
            searcher
                .search(
                    parsed.bytes(),
                    offsets,
                    table,
                    &mut matches,
                    &mut distances,
                    &mut counts,
                )
                .map_err(|error| error.to_string())?;
        } else {
            let mut scanner = ScannerAny::new(metric, &dtype, scope, std::alloc::System)
                .map_err(|error| error.to_string())?;
            scanner
                .scan(
                    parsed.bytes(),
                    offsets,
                    table,
                    &mut matches,
                    &mut distances,
                    &mut counts,
                )
                .map_err(|error| error.to_string())?;
        }
        per_shard.push(ShardResults {
            shard,
            matches,
            distances,
            counts,
        });
    }

    let nodes = tables.iter().map(GraphLike::len).sum();
    let results = merge(per_shard, &tables, parsed.count(), run.k, direction);
    let names: Vec<String> = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    let header = Header {
        metric,
        dtype: &dtype,
        shards: &names,
        nodes,
    };
    Ok(render(format, &header, first.key_dtype(), &results))
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    let outcome = match &arguments.command {
        Command::Info { index } => run_info(index),
        Command::Scan {
            index,
            query,
            metric,
            run,
        } => run_query(index, query, metric.as_deref(), run, false, 0),
        Command::Search {
            index,
            query,
            metric,
            force,
            expansion,
            run,
        } => {
            if metric.is_some() && !force {
                Err(
                    "--metric on a walk needs --force: the graph's edges were built \
                     for the declared metric, so another metric returns wrong neighbors"
                        .to_string(),
                )
            } else {
                run_query(index, query, metric.as_deref(), run, true, *expansion)
            }
        }
    };

    match outcome {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("sg: {message}");
            ExitCode::FAILURE
        }
    }
}

// endregion: Verbs

// region: Rendering

/// One ranked match, erased of every metric and dtype detail.
struct Hit {
    /// Index into the shard list a caller printed alongside, so a merged result
    /// still says which file it came from.
    shard: usize,
    /// The node's index within its own shard — the file's own addressing.
    node: u64,
    /// The node's key bytes, rendered by [`format_key`].
    key: Vec<u8>,
    /// The score, in the metric's own units, already widened.
    score: f64,
}

/// How results reach the caller. `Table` is the aligned human view, `Json` one
/// document, `Ndjson` one line per query for streaming into another tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Table,
    Json,
    Ndjson,
}

impl Format {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "table" => Format::Table,
            "json" => Format::Json,
            "ndjson" => Format::Ndjson,
            _ => return None,
        })
    }
}

/// A key rendered for display: UTF-8 when the column says so and the bytes are
/// valid, a little-endian integer for the 4- and 8-byte widths, hex otherwise.
/// Keys are opaque in the format, so this guesses only from the declared dtype.
fn format_key(key: &[u8], key_dtype: &str) -> String {
    match key_dtype {
        "utf8" => String::from_utf8_lossy(key).into_owned(),
        "u64" | "i64" if key.len() == 8 => {
            u64::from_le_bytes(key.try_into().unwrap_or_default()).to_string()
        }
        "u32" | "i32" if key.len() == 4 => {
            u32::from_le_bytes(key.try_into().unwrap_or_default()).to_string()
        }
        _ => key.iter().fold(String::new(), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        }),
    }
}

/// A score column wide enough to read: geospatial meters and RMSD want fixed
/// decimals, a dot product wants more precision, and an edit distance is an
/// integer. One rule covers all three — trailing zeros are trimmed.
fn format_score(score: f64) -> String {
    if score == score.trunc() && score.abs() < 1e15 {
        return format!("{score:.0}");
    }
    let text = format!("{score:.6}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// A score as a JSON number token. RFC 8259 has no `NaN` or `Infinity`
/// literal, and Rust's `Display` would emit those bare words — output a strict
/// parser rejects. A non-finite score can only come from a degenerate input, so
/// it renders as `null` rather than as text that is not JSON at all.
fn json_score(score: f64) -> String {
    if score.is_finite() {
        score.to_string()
    } else {
        "null".to_string()
    }
}

/// Escape a string for a JSON scalar. Keys can hold anything, so this is not
/// optional — the library's own writer is for headers, not results.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", control as u32);
            }
            other => out.push(other),
        }
    }
    out
}

/// The header a `Table` render opens with: what ran, over what.
struct Header<'a> {
    metric: &'a str,
    dtype: &'a str,
    shards: &'a [String],
    nodes: usize,
}

/// Render `results` — one `Vec<Hit>` per query, already ranked — in `format`.
fn render(format: Format, header: &Header<'_>, key_dtype: &str, results: &[Vec<Hit>]) -> String {
    match format {
        Format::Table => render_table(header, key_dtype, results),
        Format::Json => render_json(header, key_dtype, results),
        Format::Ndjson => render_ndjson(key_dtype, results),
    }
}

fn render_table(header: &Header<'_>, key_dtype: &str, results: &[Vec<Hit>]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "metric {} · dtype {} · {} node(s) · {} shard(s)",
        header.metric,
        header.dtype,
        header.nodes,
        header.shards.len()
    );
    for (index, shard) in header.shards.iter().enumerate() {
        let _ = writeln!(out, "  [{index}] {shard}");
    }

    for (query, hits) in results.iter().enumerate() {
        let _ = writeln!(out, "\nquery {query} — {} match(es)", hits.len());
        if hits.is_empty() {
            continue;
        }
        // Size the key column to its content so nothing wraps and nothing is
        // padded to a fixed guess.
        let keys: Vec<String> = hits
            .iter()
            .map(|hit| format_key(&hit.key, key_dtype))
            .collect();
        // `{:<width$}` pads by character count, so the column must be measured
        // the same way — `String::len` is bytes, and would over-pad any
        // multi-byte key. Genuinely double-width glyphs still need a
        // grapheme-aware crate; this at least stops mixing units.
        let key_width = keys
            .iter()
            .map(|key| key.chars().count())
            .max()
            .unwrap_or(3)
            .max(3);
        let multi_shard = header.shards.len() > 1;
        let _ = writeln!(
            out,
            "  {:>4}  {:>10}  {:<key_width$}  {:>12}{}",
            "rank",
            "node",
            "key",
            "score",
            if multi_shard { "  shard" } else { "" }
        );
        for (rank, (hit, key)) in hits.iter().zip(&keys).enumerate() {
            let _ = writeln!(
                out,
                "  {:>4}  {:>10}  {:<key_width$}  {:>12}{}",
                rank + 1,
                hit.node,
                key,
                format_score(hit.score),
                if multi_shard {
                    format!("  [{}]", hit.shard)
                } else {
                    String::new()
                }
            );
        }
    }
    out
}

fn render_hits(out: &mut String, key_dtype: &str, hits: &[Hit]) {
    out.push('[');
    for (index, hit) in hits.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{{\"node\":{},\"key\":\"{}\",\"score\":{},\"shard\":{}}}",
            hit.node,
            escape(&format_key(&hit.key, key_dtype)),
            json_score(hit.score),
            hit.shard
        );
    }
    out.push(']');
}

fn render_json(header: &Header<'_>, key_dtype: &str, results: &[Vec<Hit>]) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"metric\":\"{}\",\"dtype\":\"{}\",\"nodes\":{},\"shards\":[",
        escape(header.metric),
        escape(header.dtype),
        header.nodes
    );
    for (index, shard) in header.shards.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(out, "\"{}\"", escape(shard));
    }
    out.push_str("],\"queries\":[");
    for (query, hits) in results.iter().enumerate() {
        if query > 0 {
            out.push(',');
        }
        let _ = write!(out, "{{\"index\":{query},\"matches\":");
        render_hits(&mut out, key_dtype, hits);
        out.push('}');
    }
    out.push_str("]}\n");
    out
}

fn render_ndjson(key_dtype: &str, results: &[Vec<Hit>]) -> String {
    let mut out = String::new();
    for (query, hits) in results.iter().enumerate() {
        let _ = write!(out, "{{\"index\":{query},\"matches\":");
        render_hits(&mut out, key_dtype, hits);
        out.push_str("}\n");
    }
    out
}

// endregion: Rendering
