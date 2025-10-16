# Lab Rush Cost Analysis

Dataset: 20 records

## Key correlations
- Pearson corr(cost_per_hour, remaining_hours) = -0.8619
- Pearson corr(cost_per_hour, initial_hours) = -0.8751
- Pearson corr(remaining_hours, initial_hours) = 0.9939

## Regression: cost_per_hour vs ln(initial_hours)
Model: $\mathrm{cost}_h = a + b \cdot \ln(T)$ where $T$ is initial duration in hours.
- a (intercept) = 8.0797
- b (slope) = -0.4121
- R^2 = 0.9266
- RMSE = 0.1288 gems/hour

## Regression: cost_per_hour vs initial_hours (linear)
- intercept = 6.6579, slope = -0.002731, R^2 = 0.7657, RMSE = 0.2301

## Multivariate regression: cost_per_hour ~ initial_hours + remaining_hours
- intercept = 6.6772, beta_init = -0.004737, beta_rem = 0.002519
- R^2 = 0.7708, RMSE = 0.2276

## rush_cost ~ remaining_hours
- slope = 5.0473, intercept = 66.0294, R^2 = 0.9945, RMSE = 45.7688

## Empirical per-hour function
Using the ln(T) fit: r(T) ≈ 8.0797 + -0.4121 * ln(T) (T in hours).

## Notes and recommendations
- The dominant predictor of gems/hour is initial total duration. Short jobs are charged more gems/hour.
- Remaining time correlates with gems/hour only because it is correlated with initial duration; its independent effect is small.
- The game likely uses a per-hour rate that depends on original duration, then multiplies by remaining hours and rounds to integer gems.