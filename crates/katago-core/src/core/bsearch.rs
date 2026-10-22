// Ported from KataGo cpp/core/bsearch.cpp, cpp/core/bsearch.h @ ba938676d7f42d70950b3a535af2466fb642008c

/// upstream: namespace BSearch (bsearch.h)
pub struct BSearch;

impl BSearch {
  /// upstream: BSearch::findFirstGt (bsearch.cpp)
  //Assumes arr is sorted.
  //Finds the first index i within [low,high) where arr[i] > x, or high if such an index does not exist.
  pub fn find_first_gt(arr: &[f64], x: f64, low: usize, high: usize) -> usize {
    if low >= high {
      return high;
    }
    let mid: usize = (low + high) / 2;
    if arr[mid] > x {
      BSearch::find_first_gt(arr, x, low, mid)
    } else {
      BSearch::find_first_gt(arr, x, mid + 1, high)
    }
  }
}

// upstream BSearch::runTests (bsearch.cpp) -> tests/bsearch.rs (port-manifest.toml notes).
