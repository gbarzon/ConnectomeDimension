#setwd("Desktop/R/dimensionality/")

library(lme4)
library(lmerTest)
library(car)

library(sjPlot)
library(ggplot2)

# Load the CSV file into an R dataframe
delta_data <- read.csv("graph_distance_delta_dim.csv")
delta_data_rescaled <- read.csv("graph_distance_delta_dim.csv")

# Rescale variables
delta_data_rescaled$graph_distance <- delta_data_rescaled$graph_distance - mean(delta_data_rescaled$graph_distance)
delta_data_rescaled$graph_distance <- delta_data_rescaled$graph_distance / sd(delta_data_rescaled$graph_distance)

delta_data_rescaled$delta_dim <- delta_data_rescaled$delta_dim - mean(delta_data_rescaled$delta_dim)
delta_data_rescaled$delta_dim <- delta_data_rescaled$delta_dim / sd(delta_data_rescaled$delta_dim)

# Fit lmer
model <- lmer(delta_dim ~ graph_distance + (1+graph_distance| sub), data = delta_data_rescaled)
result <- summary(model)

# Print model summary
print(summary(model))
print(car::Anova(model,type=3,test="F"))

# Plot effects
library(effects)
plot(allEffects(model))

plot_model(model, type = "re", terms = c("Subj"))