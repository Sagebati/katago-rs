// Ported from KataGo cpp/core/fancymath.cpp, cpp/core/fancymath.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// Not ported (port-manifest.toml notes): FancyMath::runTests and its static testApproxEq /
// APPROX_EQ are consumed as a test (tests/fancymath.rs).
//
// PORT NOTE: lgamma, exp, log, pow, expm1 go through port::math (glibc 2.44 bit-identical on
// every target, PORTING.md §6 item 3); sqrt is correctly rounded everywhere and stays std.

use crate::port::math;

/// upstream: namespace FancyMath (fancymath.h)
pub struct FancyMath;

/// upstream: PI (fancymath.cpp) static
const PI: f64 = 3.1415926535897932384626433832795;

/// upstream: evaluateContinuedFractionHelper (fancymath.cpp) static
//Modified Lentz Algorithm
fn evaluate_continued_fraction_helper(
  numer: &mut dyn FnMut(i32) -> f64,
  denom: &mut dyn FnMut(i32) -> f64,
  tolerance: f64,
  max_terms: i32,
) -> f64 {
  let tiny: f64 = 1e-300;
  let mut ret: f64 = denom(0);
  if ret == 0.0 {
    ret = tiny;
  }
  let mut c: f64 = ret;
  let mut d: f64 = 0.0;

  let mut n: i32 = 1;
  while n < max_terms {
    let nextnumer: f64 = numer(n);
    let nextdenom: f64 = denom(n);
    d = nextdenom + nextnumer * d;
    if d == 0.0 {
      d = tiny;
    }
    c = nextdenom + nextnumer / c;
    if c == 0.0 {
      c = tiny;
    }
    d = 1.0 / d;
    let mult: f64 = c * d;
    ret = ret * mult;
    if (mult - 1.0).abs() <= tolerance {
      break;
    }
    n += 1;
  }
  ret
}

/// upstream: incompleteBetaContinuedFraction (fancymath.cpp) static
//Textbook continued fraction term for incomplete beta function
fn incomplete_beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
  let mut numer = |n: i32| -> f64 {
    if n % 2 == 0 {
      let m: f64 = (n / 2) as f64;
      m * (b - m) * x / (a + 2.0 * m - 1.0) / (a + 2.0 * m)
    } else {
      let m: f64 = ((n - 1) / 2) as f64;
      -(a + m) * (a + b + m) * x / (a + 2.0 * m) / (a + 2.0 * m + 1.0)
    }
  };
  let mut denom = |n: i32| -> f64 {
    let _ = n;
    1.0
  };
  evaluate_continued_fraction_helper(&mut numer, &mut denom, 1e-15, 100000)
}

impl FancyMath {
  /// upstream: FancyMath::beta (fancymath.cpp)
  //https://en.wikipedia.org/wiki/Beta_function
  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  pub fn beta(a: f64, b: f64) -> f64 {
    math::exp(math::lgamma(a) + math::lgamma(b) - math::lgamma(a + b))
  }

  /// upstream: FancyMath::logbeta (fancymath.cpp)
  pub fn logbeta(a: f64, b: f64) -> f64 {
    math::lgamma(a) + math::lgamma(b) - math::lgamma(a + b)
  }

  /// upstream: FancyMath::evaluateContinuedFraction (fancymath.cpp)
  pub fn evaluate_continued_fraction(
    numer: &mut dyn FnMut(i32) -> f64,
    denom: &mut dyn FnMut(i32) -> f64,
    tolerance: f64,
    max_terms: i32,
  ) -> f64 {
    evaluate_continued_fraction_helper(numer, denom, tolerance, max_terms)
  }

  /// upstream: FancyMath::incompleteBeta (fancymath.cpp)
  //https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
  pub fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if !(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0) {
      return f64::NAN;
    }
    if x <= 0.0 {
      return 0.0;
    }
    if x >= 1.0 {
      return FancyMath::beta(a, b);
    }
    let logx: f64 = math::ln(x);
    let logy: f64 = math::ln(1.0 - x);
    if x <= (a + 1.0) / (a + b + 2.0) {
      math::exp(logx * a + logy * b) / a / incomplete_beta_continued_fraction(x, a, b)
    } else {
      FancyMath::beta(a, b) - (math::exp(logy * b + logx * a) / b / incomplete_beta_continued_fraction(1.0 - x, b, a))
    }
  }

  /// upstream: FancyMath::regularizedIncompleteBeta (fancymath.cpp)
  //https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
  pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if !(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0) {
      return f64::NAN;
    }
    if x <= 0.0 {
      return 0.0;
    }
    if x >= 1.0 {
      return 1.0;
    }
    let logx: f64 = math::ln(x);
    let logy: f64 = math::ln(1.0 - x);
    if x <= (a + 1.0) / (a + b + 2.0) {
      math::exp(logx * a + logy * b - FancyMath::logbeta(a, b)) / a / incomplete_beta_continued_fraction(x, a, b)
    } else {
      1.0 - (math::exp(logy * b + logx * a - FancyMath::logbeta(a, b)) / b / incomplete_beta_continued_fraction(1.0 - x, b, a))
    }
  }

  /// upstream: FancyMath::tdistpdf (fancymath.cpp)
  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  pub fn tdistpdf(x: f64, degrees_of_freedom: f64) -> f64 {
    let v: f64 = degrees_of_freedom;
    if !(v > 0.0) {
      return f64::NAN;
    }
    1.0 / (v * PI).sqrt()
      / math::exp(math::lgamma(v / 2.0) - math::lgamma((v + 1.0) / 2.0))
      / math::powf(1.0 + x * x / v, (v + 1.0) / 2.0)
  }

  /// upstream: FancyMath::tdistcdf (fancymath.cpp)
  pub fn tdistcdf(x: f64, degrees_of_freedom: f64) -> f64 {
    let v: f64 = degrees_of_freedom;
    if !(v > 0.0) {
      return f64::NAN;
    }
    if x >= 0.0 {
      1.0 - FancyMath::regularized_incomplete_beta(v / (x * x + v), v / 2.0, 0.5) / 2.0
    } else {
      FancyMath::regularized_incomplete_beta(v / (x * x + v), v / 2.0, 0.5) / 2.0
    }
  }

  /// upstream: FancyMath::betapdf (fancymath.cpp)
  pub fn betapdf(x: f64, a: f64, b: f64) -> f64 {
    if !(x >= 0.0 && x <= 1.0 && a > 0.0 && b > 0.0) {
      return f64::NAN;
    }
    if x == 0.0 {
      return if a < 1.0 {
        f64::INFINITY
      } else if a > 1.0 {
        0.0
      } else {
        1.0 / FancyMath::beta(a, b)
      };
    }
    if x == 1.0 {
      return if b < 1.0 {
        f64::INFINITY
      } else if b > 1.0 {
        0.0
      } else {
        1.0 / FancyMath::beta(a, b)
      };
    }
    math::exp(-FancyMath::logbeta(a, b) + math::ln(x) * (a - 1.0) + math::ln(1.0 - x) * (b - 1.0))
  }

  /// upstream: FancyMath::betacdf (fancymath.cpp)
  pub fn betacdf(x: f64, a: f64, b: f64) -> f64 {
    FancyMath::regularized_incomplete_beta(x, a, b)
  }

  /// upstream: FancyMath::normToTApprox (fancymath.cpp)
  //Given z, compute and return an approximation for the value t
  //such that the probability that a draw from StudentT(degreesOfFreedom) > t
  //is the same as a probability that a draw from StandardNormal() > z
  pub fn norm_to_t_approx(z: f64, degrees_of_freedom: f64) -> f64 {
    let n: f64 = degrees_of_freedom;
    (n * (math::expm1(z * z * (n - 1.5) / ((n - 1.0) * (n - 1.0))))).sqrt()
  }

  /// upstream: FancyMath::binaryCrossEntropy (fancymath.cpp)
  //predProb is scaled into the range [epsilon,1.0-epsilon].
  pub fn binary_cross_entropy(mut pred_prob: f64, target_prob: f64, epsilon: f64) -> f64 {
    let mut reverse_prob: f64 = 1.0 - pred_prob;
    pred_prob = epsilon * (1.0 - pred_prob) + (1.0 - epsilon) * pred_prob;
    reverse_prob = epsilon * (1.0 - reverse_prob) + (1.0 - epsilon) * reverse_prob;

    // Just in case there is float weirdness
    if pred_prob < epsilon {
      pred_prob = epsilon;
    }
    if pred_prob > 1.0 - epsilon {
      pred_prob = 1.0 - epsilon;
    }
    if reverse_prob < epsilon {
      reverse_prob = epsilon;
    }
    if reverse_prob > 1.0 - epsilon {
      reverse_prob = 1.0 - epsilon;
    }

    target_prob * (-math::ln(pred_prob)) + (1.0 - target_prob) * (-math::ln(reverse_prob))
  }
}
