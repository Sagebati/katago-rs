// Ported from KataGo cpp/search/distributiontable.cpp, cpp/search/distributiontable.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use crate::port::cpp::static_cast_i32_f64;

/// upstream: DistributionTable (distributiontable.h)
/// PORT NOTE: copy constructor and assignment are deleted upstream, so no Clone. `pdfTable` and
/// `cdfTable` (`new double[size]`, freed in the destructor) are `Box<[f64]>` (OWNERSHIP.tsv).
pub struct DistributionTable {
  pub pdf_table: Box<[f64]>,
  pub cdf_table: Box<[f64]>,
  pub size: i32,
  pub min_z: f64,
  pub max_z: f64,
}

impl DistributionTable {
  /// upstream: DistributionTable::DistributionTable (distributiontable.cpp)
  pub fn new(pdf: &dyn Fn(f64) -> f64, cdf: &dyn Fn(f64) -> f64, minz: f64, maxz: f64, sz: i32) -> DistributionTable {
    let size: i32 = sz;
    let min_z: f64 = minz;
    let max_z: f64 = maxz;
    let mut pdf_table: Box<[f64]> = vec![0.0f64; size as usize].into_boxed_slice();
    let mut cdf_table: Box<[f64]> = vec![0.0f64; size as usize].into_boxed_slice();

    let mut i: i32 = 0;
    while i < size {
      if i == 0 {
        pdf_table[i as usize] = 0.0;
        cdf_table[i as usize] = 0.0;
      } else if i == size - 1 {
        pdf_table[i as usize] = 0.0;
        cdf_table[i as usize] = 1.0;
      } else {
        let z: f64 = min_z + i as f64 * (max_z - min_z) / ((size - 1) as f64);
        pdf_table[i as usize] = pdf(z);
        cdf_table[i as usize] = cdf(z);
      }
      i += 1;
    }

    DistributionTable { pdf_table, cdf_table, size, min_z, max_z }
  }

  // upstream: DistributionTable::~DistributionTable (distributiontable.cpp)
  // PORT NOTE: only frees the tables (Box<[f64]> drops them), omitted.

  /// upstream: DistributionTable::getPdfCdf (distributiontable.h)
  /// PORT NOTE: `(int)d` is C++ UB for NaN or d >= 2^31; `static_cast_i32_f64` gives x86-64's INT_MIN,
  /// after which upstream reads `pdfTable[INT_MIN]` out of bounds (UB) and Rust panics on the index.
  /// TODO(port): upstream UB (out-of-bounds table read for NaN / huge z); no upstream caller is
  /// known to reach it, so the Rust panic is left rather than inventing a result (PORTING.md §6b).
  #[inline]
  pub fn get_pdf_cdf(&self, z: f64, pdf: &mut f64, cdf: &mut f64) {
    let d: f64 = ((self.size - 1) as f64) * (z - self.min_z) / (self.max_z - self.min_z);
    if d <= 0.0 {
      *pdf = 0.0;
      *cdf = 0.0;
      return;
    }
    let idx: i32 = static_cast_i32_f64(d);
    if idx >= self.size - 1 {
      *pdf = 0.0;
      *cdf = 1.0;
      return;
    }
    let lambda: f64 = d - idx as f64;
    let yp0: f64 = self.pdf_table[idx as usize];
    let yp1: f64 = self.pdf_table[(idx + 1) as usize];
    let yc0: f64 = self.cdf_table[idx as usize];
    let yc1: f64 = self.cdf_table[(idx + 1) as usize];
    *pdf = yp0 + lambda * (yp1 - yp0);
    *cdf = yc0 + lambda * (yc1 - yc0);
  }

  /// upstream: DistributionTable::getPdf (distributiontable.h)
  /// PORT NOTE: `(int)d` UB as in getPdfCdf.
  #[inline]
  pub fn get_pdf(&self, z: f64) -> f64 {
    let d: f64 = ((self.size - 1) as f64) * (z - self.min_z) / (self.max_z - self.min_z);
    if d <= 0.0 {
      return 0.0;
    }
    let idx: i32 = static_cast_i32_f64(d);
    if idx >= self.size - 1 {
      return 0.0;
    }
    let lambda: f64 = d - idx as f64;
    let y0: f64 = self.pdf_table[idx as usize];
    let y1: f64 = self.pdf_table[(idx + 1) as usize];
    y0 + lambda * (y1 - y0)
  }

  /// upstream: DistributionTable::getCdf (distributiontable.h)
  /// PORT NOTE: `(int)d` UB as in getPdfCdf.
  #[inline]
  pub fn get_cdf(&self, z: f64) -> f64 {
    let d: f64 = ((self.size - 1) as f64) * (z - self.min_z) / (self.max_z - self.min_z);
    if d <= 0.0 {
      return 0.0;
    }
    let idx: i32 = static_cast_i32_f64(d);
    if idx >= self.size - 1 {
      return 1.0;
    }
    let lambda: f64 = d - idx as f64;
    let y0: f64 = self.cdf_table[idx as usize];
    let y1: f64 = self.cdf_table[(idx + 1) as usize];
    y0 + lambda * (y1 - y0)
  }
}
