# Advancecd System Lab Project

## Changes
* Using BunchKaufman Linear solver instead of LU decomposition
* Change the matrix to 1d array
* Change `memove` to `memcpy`
* Batch evaluation (remove procedure call)
* evaluation and build(sharing of common subexpressions)
    * reuse `sqrt`
    * reuse index    
