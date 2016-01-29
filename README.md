# Ensemble pdf creation for temperature
[![Build Status](https://travis-ci.org/tomderuijter/probability-forecast.svg?branch=master)](https://travis-ci.org/tomderuijter/probability-forecast)

## Verification
- Rank histograms
- Skill scores
- CRPS
- QQ plot

## Algorithms for single forecast hour

**Assumptions**
* Temperatures can be modeled by Normal distributions
* Ensemble members are equally likely to represent the observation a priori
* The ensemble set behaves as a randomly selected sample from the expected
distribution of observations.

**Pseudo code**
```
function main() {
Input:
    Forecast hours F
    ensemble set S
Output:
    PDF for each forecast hour

For each forecast hour f in F:
    For each ensemble member m in set s:
        m.variance <- getVariance()
    f.pdf <- getPdf()

return F
}
```

**Options for variance**
* Get control member variance
* Derive member variance from ensemble mean variance

**Options for model combination**
* Uniform distribution as prior

## Data
The following data columns
* Model name + Element name
* Element observation value
* Element value
* Station id
* Forecast hour
* Issue date
* Valid date

## Ideas
Do something with covariance between model hours
- Maybe Gaussian process fit on the ensemble members.
- Or fit all hours together respecting their covariance
- Maybe have a look into time series modeling to see if we can use autocorrelation

Relate variance between multiple issues for the same forecast hour
