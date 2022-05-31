# Advancecd System Lab Project

## Changes
* Using BunchKaufman Linear solver instead of LU decomposition
* Change the matrix to 1d array
* Change `memove` to `memcpy`
* Batch evaluation (remove procedure call)
* evaluation and build(sharing of common subexpressions)
    * reuse `sqrt`
    * reuse index   

performance plot:
1a. evaluate surrogate - static replacement, remove dependency and unrolling;
1b. evaluate surrogate - vectorization; 
2. Bunchkaufman pivoting for symmetric indefinite matrix factorization; 
3. blocking - cache hit/ cache fit; 
4a. static replacement, remove dependency and unrolling; 
4b vectorization.

flops count of Bunchkaufman, evaluate surrogate, blocking. 
