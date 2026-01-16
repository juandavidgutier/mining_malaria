library(EValue)

# Read data
data <- read.csv("D:/param_evalue_mining_final.csv")

# Display data
cat(rep("=", 80), "\n")
cat("Data loaded:\n")
print(data[, c('Analysis', 'RR_point_estimate', 'RR_lower_CI', 'RR_upper_CI')])

# IMPORTANT: Verify that RR_lower_CI < RR_upper_CI
if (data$RR_lower_CI > data$RR_upper_CI) {
  cat("\n⚠️ WARNING: Confidence interval bounds are inverted. Swapping them...\n")
  temp <- data$RR_lower_CI
  data$RR_lower_CI <- data$RR_upper_CI
  data$RR_upper_CI <- temp
}

# Calculate E-value for Risk Ratio
cat("\n", rep("=", 80), "\n")
cat("E-VALUE CALCULATION\n")
cat(rep("=", 80), "\n\n")

evalue_result <- evalues.RR(
  est = data$RR_point_estimate,
  lo = data$RR_lower_CI,
  hi = data$RR_upper_CI,
  true = 1  # Null hypothesis: RR = 1 (no effect)
)

print(evalue_result)

# Interpretation
cat("\n", rep("=", 80), "\n")
cat("E-VALUE INTERPRETATION\n")
cat(rep("=", 80), "\n\n")

cat("The E-value represents the minimum strength of association that an\n")
cat("unmeasured confounder would need to have (simultaneously with both the exposure\n")
cat("and the outcome) to fully explain away the observed effect.\n\n")

cat("E-value for the point estimate:", round(evalue_result[1, "E-values"], 2), "\n")
cat("E-value for the confidence interval limit:", round(evalue_result[2, "E-values"], 2), "\n\n")

if (data$RR_point_estimate > 1) {
  cat("Since RR > 1, we look for confounders that INCREASE the risk.\n")
} else if (data$RR_point_estimate < 1) {
  cat("Since RR < 1, we look for confounders that DECREASE the risk.\n")
}

cat(rep("=", 80), "\n")