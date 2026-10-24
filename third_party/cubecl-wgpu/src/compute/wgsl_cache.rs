//! PATCH(katago-rs): a WGSL kernel cache the embedder fills and drains, for targets where cubecl
//! persists no compiled kernel (wasm: WGSL "is compiled by the driver on every run", so upstream
//! only stores SPIR-V). Without it every page load expands, optimizes and prints every kernel
//! again, which is most of a WebGPU warm-up in the browser (~1 s for KataGo's b10c128).
//!
//! Off until [`enable`] is called; then [`crate::WgpuServer`] records each WGSL kernel it compiles
//! and, before compiling, looks the kernel up here. Entries are keyed like cubecl's own persistent
//! SPIR-V store (device properties hash, kernel id stable hash) and also carry the kernel id's
//! [stable format](cubecl_server::id::KernelId::stable_format), which must match exactly for a hit.
//! As with that store, an entry is only valid for the build that wrote it: the embedder must keep
//! blobs per build (and per adapter). [`export`] / [`import`] move the whole cache as one blob.

use cubecl_common::hash::StableHash;
use cubecl_core::prelude::Visibility;
use cubecl_server::id::KernelId;
use cubecl_server::kernel::BufferIOAttr;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

/// A compiled WGSL kernel: what [`crate::WgpuServer`] needs to build its pipeline without compiling.
#[derive(Debug)]
pub(crate) struct Entry {
    pub stable_format: String,
    pub entrypoint_name: String,
    pub source: String,
    pub buffers: Vec<Visibility>,
    pub io: Vec<BufferIOAttr>,
    pub shared_memory_size: usize,
}

#[derive(Default)]
struct State {
    enabled: bool,
    entries: HashMap<(u64, StableHash), Arc<Entry>>,
    /// Kernels compiled since the last [`export`].
    added: usize,
    hits: u64,
    misses: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(State::default))
}

/// Turns the cache on (lookups and recording).
pub fn enable() {
    with(|s| s.enabled = true);
}

/// `(entries, kernels compiled since the last export, hits, misses)`.
pub fn stats() -> (usize, usize, u64, u64) {
    with(|s| (s.entries.len(), s.added, s.hits, s.misses))
}

pub(crate) fn is_enabled() -> bool {
    with(|s| s.enabled)
}

pub(crate) fn get(properties_hash: u64, id: &KernelId) -> Option<Arc<Entry>> {
    with(|s| {
        if !s.enabled {
            return None;
        }
        let found = s
            .entries
            .get(&(properties_hash, id.stable_hash()))
            .filter(|e| e.stable_format == id.stable_format())
            .cloned();
        if found.is_some() {
            s.hits += 1;
        } else {
            s.misses += 1;
        }
        found
    })
}

pub(crate) fn insert(properties_hash: u64, id: &KernelId, entry: Entry) {
    with(|s| {
        if s.enabled {
            s.entries
                .insert((properties_hash, id.stable_hash()), Arc::new(entry));
            s.added += 1;
        }
    })
}

const MAGIC: &[u8; 8] = b"CCLWGSL2";

/// The whole cache as one blob (for [`import`] in a later process of the same build).
pub fn export() -> Vec<u8> {
    with(|s| {
        s.added = 0;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        put_u32(&mut out, s.entries.len() as u32);
        for (&(props, hash), e) in &s.entries {
            out.extend_from_slice(&props.to_le_bytes());
            out.extend_from_slice(&hash.to_le_bytes());
            put_str(&mut out, &e.stable_format);
            put_str(&mut out, &e.entrypoint_name);
            put_str(&mut out, &e.source);
            put_u32(&mut out, e.buffers.len() as u32);
            out.extend(e.buffers.iter().map(|v| match v {
                Visibility::Uniform => 0u8,
                Visibility::Read => 1,
                Visibility::ReadWrite => 2,
            }));
            put_u32(&mut out, e.io.len() as u32);
            out.extend(e.io.iter().map(|v| match v {
                BufferIOAttr::ReadOnly => 0u8,
                BufferIOAttr::WriteOnly => 1,
                BufferIOAttr::ReadWrite => 2,
                BufferIOAttr::Dead => 3,
            }));
            out.extend_from_slice(&(e.shared_memory_size as u64).to_le_bytes());
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    })
}

/// 64-bit FNV-1a: catches a truncated or damaged blob (whose WGSL would otherwise reach the driver).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Adds the entries of an [`export`]ed blob; returns how many it held. A malformed blob adds nothing.
pub fn import(bytes: &[u8]) -> Result<usize, String> {
    let entries = parse(bytes).ok_or_else(|| "malformed WGSL cache blob".to_string())?;
    let n = entries.len();
    with(|s| {
        for (key, e) in entries {
            s.entries.entry(key).or_insert_with(|| Arc::new(e));
        }
    });
    Ok(n)
}

fn parse(b: &[u8]) -> Option<Vec<((u64, StableHash), Entry)>> {
    let (b, checksum) = b.split_at_checked(b.len().checked_sub(8)?)?;
    if fnv1a(b) != u64::from_le_bytes(checksum.try_into().ok()?) {
        return None;
    }
    let mut r = Reader { b, i: 0 };
    if r.take(8)? != MAGIC {
        return None;
    }
    let n = r.u32()? as usize;
    let mut v = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        let props = u64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let hash = StableHash::from_le_bytes(r.take(16)?.try_into().ok()?);
        let stable_format = r.str()?;
        let entrypoint_name = r.str()?;
        let source = r.str()?;
        let nb = r.u32()? as usize;
        let buffers = r
            .take(nb)?
            .iter()
            .map(|x| match x {
                0 => Some(Visibility::Uniform),
                1 => Some(Visibility::Read),
                2 => Some(Visibility::ReadWrite),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let ni = r.u32()? as usize;
        let io = r
            .take(ni)?
            .iter()
            .map(|x| match x {
                0 => Some(BufferIOAttr::ReadOnly),
                1 => Some(BufferIOAttr::WriteOnly),
                2 => Some(BufferIOAttr::ReadWrite),
                3 => Some(BufferIOAttr::Dead),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let shared_memory_size = u64::from_le_bytes(r.take(8)?.try_into().ok()?) as usize;
        v.push((
            (props, hash),
            Entry {
                stable_format,
                entrypoint_name,
                source,
                buffers,
                io,
                shared_memory_size,
            },
        ));
    }
    (r.i == b.len()).then_some(v)
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.i..self.i.checked_add(n)?)?;
        self.i += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
}

fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

