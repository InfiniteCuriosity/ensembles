
# 1. Define the problem ---------------------------------------------------
# Determine the most accurate model to predict all employees in the United States, and
# Use that model to make predictions for the next three months

# 2. Load the R packages in alphabetical order ----------------------------

library(blsR) # To retrieve Bureau of Labor Statistics data
library(fpp3) # The gold mine of time series functions!
library(gt) # For making beautiful tables and reports
library(gtExtras) # For making beautiful reports
library(gtools) # For making beautiful reports
library(tidyverse) # My favorite way to do data science


# 3. Load the data set, convert to tsibble, plot data visualizations ------

Time_Series_data <- blsR::get_n_series(series_ids = "CES0000000001",start_year = 2021, end_year = 2023)
Time_Series_table <- data_as_tidy_table(Time_Series_data$CES0000000001$data)
Time_Series <- Time_Series_table %>% 
  mutate(Year = as.character(year),
         Month = as.character(month.abb[Time_Series_table$month]),
         Date1 = paste(Year, Month),
         Date = yearmonth(Date1),
         Value = value,
         Value_Difference = difference(Value)
  ) %>% 
  dplyr::select(Date, Value, Value_Difference) %>% 
  as_tsibble(index = Date)

Time_Series <- Time_Series[2:nrow(Time_Series),] # removes the first row, it has an NA in it from the difference calculation

sum(is.na(Time_Series)) # Check for any missing data, which will need to be addressed before analysis can move forward.

# <----- Plot of Value ----------------------------------------------> ####

Time_Series %>% 
  autoplot(Value) +
  ggtitle("Number of all value by month") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Total value ('000)")

# <----- Table of Value ---------------------------------------------> ####

gt::gt(tail(Time_Series[,c(1:3)]),
  caption = "Total value, by month") %>% 
  gt::fmt_number(columns = c('Value'), decimals = 0, use_seps = TRUE)

# <----- Plot of Trend of the Value ---------------------------------> ####
Time_Series_decomposition <- Time_Series %>% 
  model(stl = STL(Value)) %>% 
  components()

Time_Series_decomposition %>% 
  as_tsibble() %>% 
  autoplot(Value) +
  geom_line(aes(y = trend), color = "red") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    y = "Total number of value",
    title = "Total value, with trend (in red)"
  )

# <----- Plot Current Value vs Seasonally Adjusted-> ####

Time_Series_decomposition %>% 
  as_tsibble() %>% 
  autoplot(Value) +
  geom_line(aes(y = season_adjust), color = "red") +
  labs(
    y = "Total number of value, and seasonally adjusted values",
    title = "Total value, with seasonally adjusted value (in red)"
  )

# <----- Plot of Decomposition of the Value--------------------------> ####

Time_Series_decomposition %>% 
  autoplot()

# <----- Plot of Anomalies of the Value------------------------------> ####

remainder = Time_Series_decomposition$remainder

remainder1 <- sd(remainder)

Time_Series_Anomalies <- ggplot(data = Time_Series_decomposition, aes(x = Date, y = remainder)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = c(remainder1, -remainder1), linetype = 'dashed', color = 'blue') +
  geom_hline(yintercept = c(2*remainder1, -2*remainder1), linetype = 'dashed', color = 'red') +
  geom_hline(yintercept = 0, color = 'black') +
  labs(title = "Anomalies in value data \nblue line = 1 standard deviation +/- 0, red line = 2 standard deviations +/- 0")

Time_Series_Anomalies

# 4. Model and forecasts for each model -----------------------------------

# <--- 34 individual models and seven ensembles to predict Value ----> ####

Time_Series_train <- Time_Series[1:round(0.6*nrow(Time_Series)), ]
Time_Series_test <- Time_Series[nrow(Time_Series_train) +1 : (nrow(Time_Series )- nrow(Time_Series_train)), ]

fit <- Time_Series_train %>% 
  model(
    Linear1 = TSLM(Value ~ season() + trend()),
    Linear2 = TSLM(Value),
    Linear3 = TSLM(Value ~ season()),
    Linear4 = TSLM(Value ~ trend()),
    Arima1 = ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima2 = ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima3 = ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima4 = ARIMA(Value),
    Deterministic = ARIMA(Value ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Stochastic = ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Ets1 = ETS(Value ~ season() + trend()),
    Ets2 = ETS(Value ~ trend()),
    Ets3 = ETS(Value ~ season()),
    Ets4 = ETS(Value),
    Holt_Winters_Additive = ETS(Value ~ error("A") + trend("A") + season("A")),
    Holt_Winters_Multiplicative = ETS(Value ~ error("M") + trend("A") + season("M")),
    Holt_Winters_Damped = ETS(Value ~ error("M") + trend("Ad") + season("M")),
    Fourier1 = ARIMA(log(Value) ~ fourier(K = 1) + PDQ(0,0,0)),
    Fourier2 = ARIMA(log(Value) ~ fourier(K = 2) + PDQ(0,0,0)),
    Fourier3 = ARIMA(log(Value) ~ fourier(K = 3) + PDQ(0,0,0)),
    Fourier4 = ARIMA(log(Value) ~ fourier(K = 4) + PDQ(0,0,0)),
    Fourier5 = ARIMA(log(Value) ~ fourier(K = 5) + PDQ(0,0,0)),
    Fourier6 = ARIMA(log(Value) ~ fourier(K = 6) + PDQ(0,0,0)),
    Prophet_Additive = fable.prophet::prophet(Value ~ season(period = 12, type = "additive")),
    Prophet_Multiplicative = fable.prophet::prophet(Value ~ season(period = 12, type = "multiplicative")),
    NeuralNet1 = fable::NNETAR(Value),
    NeuralNet2 = fable::NNETAR(Value ~ season()),
    NeuralNet3 = fable::NNETAR(Value ~ trend()),
    NeuralNet4 = fable::NNETAR(Value ~ season() + trend()),
    VAR1 = VAR(Value),
    Mean = fable::MEAN(Value),
    Naive = fable::NAIVE(Value),
    SNaive = fable::SNAIVE(Value),
    Drift = fable::SNAIVE(Value ~ drift())
  ) %>% 
  mutate(
    Ensemble_Linear = (Linear1 + Linear2 + Linear3 + Linear4)/4,
    Ensemble_Arima = (Arima1 + Arima2 + Arima3 + Arima4 + Deterministic + Stochastic) / 6,
    Ensemble_Ets = (Ets1 + Ets2 + Ets3 + Ets4) /4,
    Ensemble_Holt_Winters = (Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped) / 3,
    Ensemble_Fourier = (Fourier1 + Fourier2 + Fourier3 + Fourier4 + Fourier5 + Fourier6) / 6,
    Ensemble = (Linear1 + Linear2 + Linear3 + Linear4 + Arima1 + Arima2 + Arima3 + Arima4 + Deterministic + Stochastic +
                  Ets1 + Ets2 + Ets3 + Ets4 + Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped +
                  Fourier1 + Fourier2 + Fourier3 + Fourier4 + Fourier5 + Fourier6 + Prophet_Additive + Prophet_Multiplicative +
                  NeuralNet1 + NeuralNet2 + NeuralNet3 + NeuralNet4 + VAR1 + Mean + Naive + SNaive + Drift) / 34
  )

Value_forecast_accuracy <- fit %>% 
  forecast(h = 3) %>% 
  accuracy(Time_Series) %>% 
  select(.model:RMSSE) %>% 
  arrange(RMSE)

gt::gt(Value_forecast_accuracy, caption = "Time series forecast accuracy, sorted by Root Mean Squared Error (RMSE)")


# 5. Calculate sim to use results for bagged forecasts --------------------

sim <- Time_Series %>% 
  model(stl = STL(Value)) %>% 
  generate(new_data = Time_Series, times = 10, bootstrap_block_size = 8) %>% 
  dplyr::select(-.model, -Value)


# 6. Diagnostics plots of the best model for value ------------------------

# Arima1 Results ----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Arima1"){
  
  Best_Model <- Time_Series %>% 
    model(
      Arima1 = ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Arima 1 model three month forecast of value")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total value")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Arima 1 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Arima 1 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Arima 1 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Arima1 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Arima1 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Arima1 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Arima1 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally Adjusted value")
  
  Arima_1_Bagged_forecasts <- sim %>% 
    model(
      Arima1 = ARIMA(.sim ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Arima_1_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Arima2 Results ----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Arima2"){
  
  Best_Model <- Time_Series %>% 
    model(
      Arima2 = ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    )

  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series) +
    labs(title = "Arima 2 model three month forecast of value")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Value")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Arima 2 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Arima 2 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Arima 2 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Arima2 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Arima2 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Arima2 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Arima2 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally Adjusted value")
  
  Arima_2_Bagged_forecasts <- sim %>% 
    model(
      Arima2 = ARIMA(.sim ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Arima_2_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Arima3 Results ----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Arima3"){
  
  Best_Model <- Time_Series %>% 
    model(
      Arima3 = ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Arima 3 model three month forecast of value")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total value")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Arima 3 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Arima 3 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Arima 3 histogram of residuals")

  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Arima3 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Arima3 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Arima3 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Arima3 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Arima_3_Bagged_forecasts <- sim %>% 
    model(
      Arima3 = ARIMA(.sim ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Arima_3_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Arima4 Results ----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Arima4"){
  
  Best_Model <- Time_Series %>% 
    model(
      Arima4 = ARIMA(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Arima 4 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Arima 4 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Arima 4 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Arima 4 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Arima4 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Arima4 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Arima4 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Arima4 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Arima_4_Bagged_forecasts <- sim %>% 
    model(
      Arima4 = ARIMA(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Arima_4_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Deterministic Results ---------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Deterministic"){
  
  Best_Model <- Time_Series %>% 
    model(
      Deterministic = ARIMA(Value ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Deterministic model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Deterministic innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Deterministic Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Deterministic histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Deterministic Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Deterministic Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Deterministic Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Deterministic Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Deterministic_Bagged_forecasts <- sim %>% 
    model(
      Deterministic = ARIMA(.sim ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    ) %>% 
    forecast(h = 3) %>% 
    print(n = 100)
  
  Bagged_Summary <- Deterministic_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Ets1 Results ------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ets1"){
  
  Best_Model <- Time_Series %>% 
    model(
      Ets1 = ETS(Value ~ season() + trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "ETS 1 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Ets 1 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Ets 1 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Ets 1 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Ets1 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Ets1 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Ets1 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Ets1 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  ETS_1_Bagged_forecasts <- sim %>% 
    model(
      Ets1 = ETS(.sim ~ season() + trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- ETS_1_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Ets2 Results ------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ets2"){
  
  Best_Model <- Time_Series %>% 
    model(
      Ets2 = ETS(Value ~ trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "ETS 2 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Ets 2 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Ets 2 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Ets 2 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Ets2 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Ets2 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Ets2 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Ets2 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  ETS_2_Bagged_forecasts <- sim %>% 
    model(
      Ets2 = ETS(.sim ~ trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- ETS_2_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Ets3 Results ------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ets3"){
  
  Best_Model <- Time_Series %>% 
    model(
      Ets3 = ETS(Value ~ season())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "ETS 3 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total value")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Ets 3 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Ets 3 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Ets 3 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Ets3 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Ets3 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Ets3 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Ets3 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  ETS_3_Bagged_forecasts <- sim %>% 
    model(
      Ets3 = ETS(.sim ~ season()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- ETS_3_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Ets4 Results ------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ets4"){
  
  Best_Model <- Time_Series %>% 
    model(
      Ets4 = ETS(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "ETS 4 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Ets 4 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Ets 4 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Ets 4 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Ets4 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Ets4 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Ets4 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Ets4 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  ETS_4_Bagged_forecasts <- sim %>% 
    model(
      Ets4 = ETS(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- ETS_4_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier1 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier1"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier1 = ARIMA(log(Value) ~ fourier(K = 1) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 1 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 1 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 1 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 1 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier1 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier1 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier1 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier1 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_1_Bagged_forecasts <- sim %>% 
    model(
      Fourier1 = ARIMA(log(.sim) ~ fourier(K = 1) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_1_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier2 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier2"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier2 = ARIMA(log(Value) ~ fourier(K = 2) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 2 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 2 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 2 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 2 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier2 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier2 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier2 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier2 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_2_Bagged_forecasts <- sim %>% 
    model(
      Fourier2 = ARIMA(log(.sim) ~ fourier(K = 2) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_2_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier3 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier3"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier3 = ARIMA(log(Value) ~ fourier(K = 3) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 3 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 3 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 3 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 3 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier3 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier3 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier3 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier3 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_3_Bagged_forecasts <- sim %>% 
    model(
      Fourier3 = ARIMA(log(.sim) ~ fourier(K = 3) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_3_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier4 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier4"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier4 = ARIMA(log(Value) ~ fourier(K = 4) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 4 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 4 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 4 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 4 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier4 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier4 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier4 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier4 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_4_Bagged_forecasts <- sim %>% 
    model(
      Fourier4 = ARIMA(log(.sim) ~ fourier(K = 4) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_4_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier5 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier5"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier5 = ARIMA(log(Value) ~ fourier(K = 5) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 5 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 5 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 5 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 5 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier5 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Actual")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier5 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier5 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier5 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_5_Bagged_forecasts <- sim %>% 
    model(
      Fourier5 = ARIMA(log(.sim) ~ fourier(K = 5) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_5_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Fourier6 Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Fourier6"){
  
  Best_Model <- Time_Series %>% 
    model(
      Fourier6 = ARIMA(log(Value) ~ fourier(K = 6) + PDQ(0,0,0))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Fourier 6 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Fourier 6 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Fourier 6 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Fourier 6 histogram of residuals")  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Fourier6 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Fourier6 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Fourier6 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Fourier6 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Fourier_6_Bagged_forecasts <- sim %>% 
    model(
      Fourier6 = ARIMA(log(.sim) ~ fourier(K = 6) + PDQ(0,0,0)),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Fourier_6_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Holt-Winters Additive Results -------------------------------------------

if(Value_forecast_accuracy[1,1] == "Holt_Winters_Additive"){
  
  Best_Model <- Time_Series %>% 
    model(
      Holt_Winters_Additive = ETS(Value ~ error("A") + trend("A") + season("A"))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Holt-Winters Additive model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Holt-Winters Additive innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Holt-Winters Additive Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Holt-Winters Additive histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Holt_Winters_Additive Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Holt_Winters_Additive Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Holt_Winters_Additive Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Holt_Winters_Additive Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Holt_Winters_Additive_Bagged_forecasts <- sim %>% 
    model(
      Holt_Winters_Additive = ETS(.sim ~ error("A") + trend("A") + season("A")),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Holt_Winters_Additive_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Holt-Winters Multiplicative Results -------------------------------------

if(Value_forecast_accuracy[1,1] == "Holt_Winters_Multiplicative"){
  
  Best_Model <- Time_Series %>% 
    model(
      Holt_Winters_Multiplicative = ETS(Value ~ error("M") + trend("A") + season("M"))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Holt-Winters Multiplicative model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Holt-Winters Multiplicative innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Holt-Winters Multiplicative Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Holt-Winters Multiplicative histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Holt_Winters_Multiplicative Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Holt_Winters_Multiplicative Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Holt_Winters_Multiplicative Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Holt_Winters_Multiplicative Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")

  Holt_Winters_Multiplicative_Bagged_forecasts <- sim %>% 
    model(
      Holt_Winters_Multiplicative = ETS(.sim ~ error("M") + trend("A") + season("M")),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Holt_Winters_Multiplicative_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Holt-Winters Damped Results ---------------------------------------------

if(Value_forecast_accuracy[1,1] == "Holt_Winters_Damped"){
  
  Best_Model <- Time_Series %>% 
    model(
      Holt_Winters_Damped = ETS(Value ~ error("M") + trend("Ad") + season("M"))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Holt-Winters Damped model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Holt-Winters Damped innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Holt-Winters Damped Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Holt-Winters Damped histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Holt_Winters_Damped Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Holt_Winters_Damped Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Holt_Winters_Damped Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Holt_Winters_Damped Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Holt_Winters_Damped_Bagged_forecasts <- sim %>% 
    model(
      Holt_Winters_Damped = ETS(.sim ~ error("M") + trend("Ad") + season("M")),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Holt_Winters_Damped_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}


# Linear1 Results ---------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Linear1"){
  
  Best_Model <- Time_Series %>% 
    model(
      Linear1 = TSLM(Value ~ season() + trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Linear 1 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Linear 1 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Linear 1 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Linear 1 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Linear1 Actual vs Predicted")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Linear1 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Linear1 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Linear1 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Linear_1_Bagged_forecasts <- sim %>% 
    model(
      Linear1 = TSLM(.sim ~ season() + trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Linear_1_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Linear2 Results ---------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Linear2"){
  
  Best_Model <- Time_Series %>% 
    model(
      Linear2 = TSLM(Value ~ season() + trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Linear 2 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Linear 2 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Linear 2 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Linear 2 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Linear2 Actual vs Predicted")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Linear2 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Linear2 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Linear2 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Linear_2_Bagged_forecasts <- sim %>% 
    model(
      Linear2 = TSLM(.sim ~ season() + trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Linear_2_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}


# Linear3 Results ---------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Linear3"){
  
  Best_Model <- Time_Series %>% 
    model(
      Linear3 = TSLM(Value ~ season())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Linear 3 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Linear 3 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Linear 3 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Linear 3 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Linear3 Actual vs Predicted")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Linear3 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Linear3 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Linear3 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Linear_3_Bagged_forecasts <- sim %>% 
    model(
      Linear3 = TSLM(.sim ~ season()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Linear_3_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Linear4 Results ---------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Linear4"){
  
  Best_Model <- Time_Series %>% 
    model(
      Linear4 = TSLM(Value ~ trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Linear 4 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Linear 4 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Linear 4 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Linear 4 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Linear4 Actual vs Predicted")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Linear4 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Linear4 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Linear4 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Linear_4_Bagged_forecasts <- sim %>% 
    model(
      Linear4 = TSLM(.sim ~ trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Linear_4_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Mean Results ------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Mean"){
  
  Best_Model <- Time_Series %>% 
    model(
      Mean = fable::MEAN(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Mean model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Mean innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Mean Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Mean histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Mean Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Mean Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Mean Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Mean Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Mean_Bagged_forecasts <- sim %>% 
    model(
      Mean = fable::MEAN(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Mean_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Naive Results -----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Naive"){
  
  Best_Model <- Time_Series %>% 
    model(
      Naive = fable::NAIVE(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Naive model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[2:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Naive innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Naive Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[2:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Naive histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[2:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Naive Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[2:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Naive Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Naive Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Naive Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Naive_Bagged_forecasts <- sim %>% 
    model(
      Naive = fable::NAIVE(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Naive_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}


# NeuralNet1 Results ------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "NeuralNet1"){
  
  Best_Model <- Time_Series %>% 
    model(
      NeuralNet1 = fable::NNETAR(Value)
    )
  
  augment(Best_Model)[13:nrow(augment(Best_Model)),]
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "NeuralNet 1 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "NeuralNet 1 innovation residuals")
  
  Best_ACF <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "NeuralNet 1 Autocorrelation function")
  
  Best_Histogram_of_Residuals <-augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "NeuralNet 1 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("NeuralNet1 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("NeuralNet1 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("NeuralNet1 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("NeuralNet1 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  NeuralNet_1_Bagged_forecasts <- sim %>% 
    model(
      NeuralNet1 = fable::NNETAR(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- NeuralNet_1_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# NeuralNet2 Results ------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "NeuralNet2"){
  
  Best_Model <- Time_Series %>% 
    model(
      NeuralNet2 = fable::NNETAR(Value ~ season())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "NeuralNet 2 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "NeuralNet 2 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "NeuralNet 2 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "NeuralNet 2 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(Time_Series),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("NeuralNet2 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(Time_Series),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("NeuralNet2 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("NeuralNet2 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("NeuralNet2 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  NeuralNet_2_Bagged_forecasts <- sim %>% 
    model(
      NeuralNet2 = fable::NNETAR(.sim ~ season()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- NeuralNet_2_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# NeuralNet3 Results ------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "NeuralNet3"){
  
  Best_Model <- Time_Series %>% 
    model(
      NeuralNet3 = fable::NNETAR(Value ~ trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "NeuralNet 3 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "NeuralNet 3 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "NeuralNet 3 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "NeuralNet 3 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("NeuralNet3 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("NeuralNet3 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("NeuralNet3 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("NeuralNet3 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  NeuralNet_3_Bagged_forecasts <- sim %>% 
    model(
      NeuralNet3 = fable::NNETAR(.sim ~ trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- NeuralNet_3_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# NeuralNet4 Results ------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "NeuralNet4"){
  
  Best_Model <- Time_Series %>% 
    model(
      NeuralNet4 = fable::NNETAR(Value ~ season() + trend())
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "NeuralNet 4 model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "NeuralNet 4 innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "NeuralNet 4 Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "NeuralNet 4 histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("NeuralNet4 Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("NeuralNet4 Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("NeuralNet4 Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("NeuralNet4 Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  NeuralNet_4_Bagged_forecasts <- sim %>% 
    model(
      NeuralNet4 = fable::NNETAR(.sim ~ season() + trend()),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- NeuralNet_4_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Prophet Additive Results ------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Prophet_Additive"){
  
  Best_Model <- Time_Series %>% 
    model(
      Prophet_Additive = fable.prophet::prophet(Value ~ season(period = 12, type = "additive"))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Prophet Additive model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Prophet Additive innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Prophet Additive Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Prophet Additive histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Prophet_Additive Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Prophet_Additive Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Prophet_Additive Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Prophet_Additive Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Prophet_Additive_Bagged_forecasts <- sim %>% 
    model(
      Prophet_Additive = fable.prophet::prophet(.sim ~ season(period = 12, type = "additive")),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Prophet_Additive_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Prophet Multiplicative Results ------------------------------------------

if(Value_forecast_accuracy[1,1] == "Prophet_Multiplicative"){
  
  Best_Model <- Time_Series %>% 
    model(
      Prophet_Multiplicative = fable.prophet::prophet(Value ~ season(period = 12, type = "multiplicative"))
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Prophet Multiplicative model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Prophet Multiplicative innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Prophet Multiplicative Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Prophet Multiplicative histogram of residuals")  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Prophet_Multiplicative Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Prophet_Multiplicative Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Prophet_Multiplicative Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Prophet_Multiplicative Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Prophet_Multiplicative_Bagged_forecasts <- sim %>% 
    model(
      Prophet_Multiplicative = fable.prophet::prophet(.sim ~ season(period = 12, type = "multiplicative")),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Prophet_Multiplicative_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# SNaive Results ----------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "SNaive"){
  
  Best_Model <- Time_Series %>% 
    model(
      SNaive = fable::SNAIVE(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Seasonal Naive model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "SNaive innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "SNaive Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[13:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "SNaive histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("SNaive Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("SNaive Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("SNaive Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("SNaive Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  SNaive_Bagged_forecasts <- sim %>% 
    model(
      SNaive = fable::SNAIVE(.sim)
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- SNaive_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Stochastic Results ------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Stochastic"){
  
  Best_Model <- Time_Series %>% 
    model(
      Stochastic = ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "Stochastic model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model) %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "Stochastic innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "Stochastic Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "Stochastic histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("Stochastic Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model), mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("Stochastic Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("Stochastic Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("Stochastic Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  Stochastic_Bagged_forecasts <- sim %>% 
    model(
      Stochastic = ARIMA(.sim ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- Stochastic_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# VAR Results -------------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "VAR"){
  
  Best_Model <- Time_Series %>% 
    model(
      VAR1 = VAR(Value)
    )
  
  augment(Best_Model)
  
  Best_Forecast <- Best_Model %>% 
    forecast(h = 3)
  
  Best_Forecast_plot <- Best_Model %>% 
    forecast(h = 3) %>% 
    autoplot(Time_Series_test) +
    labs(title = "VAR model three month forecast of total values")+
    scale_y_continuous(labels = scales::comma) +
    ylab("Total values")
  
  Best_Forecast_Min <- 
    Best_Forecast$.mean - Value_forecast_accuracy[1,4]
  
  Best_Forecast_Max <- 
    Best_Forecast$.mean + Value_forecast_accuracy[1,4]
  
  Best_STL_Decomposition <- 
    Time_Series[2:nrow(Time_Series),] %>% 
    model(
      stl = STL(Value)
    ) %>% 
    components()
  
  Best_STL_Decomposition_autoplot <- Best_STL_Decomposition %>% autoplot()
  
  Best_Innovation_Residuals <- augment(Best_Model)[6:nrow(Time_Series),] %>% 
    autoplot(.innov) +
    geom_point() +
    geom_hline(yintercept = 0, color = "red") +
    labs(title = "VAR innovation residuals")
  
  Best_ACF <- augment(Best_Model) %>% 
    ACF(.innov) %>% 
    autoplot() +
    labs(title = "VAR Autocorrelation function")
  
  Best_Histogram_of_Residuals <- augment(Best_Model)[6:nrow(Time_Series),] %>% 
    ggplot(aes(x = .resid)) +
    geom_histogram(bins = round(nrow(Time_Series)/5)) +
    geom_vline(xintercept = 0, color = "red") +
    labs(title = "VAR histogram of residuals")
  
  Best_Actual_vs_Predicted <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .fitted)) +
    geom_point() +
    ggtitle("VAR Actual vs Predicted") +
    geom_abline(slope = 1, intercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Predicted")
  
  Best_Actual_vs_Residuals <- 
    ggplot(augment(Best_Model)[13:nrow(augment(Best_Model)),], mapping = aes(x = Value, y = .resid)) +
    geom_point() +
    ggtitle("VAR Actual vs Residuals") +
    geom_hline(yintercept = 0, color = "red")+
    xlab("Actual") +
    ylab("Residuals")
  
  Best_Actual_vs_Trend <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = trend)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = trend), color = "red") +
    ggtitle("VAR Actual (black) vs trend (red)")+
    xlab("Actual") +
    ylab("Trend")
  
  Best_Actual_vs_Seasonally_Adjusted <- 
    ggplot(Best_STL_Decomposition, mapping = aes(x = Value, y = season_adjust)) +
    geom_line(mapping = aes(y = Value)) +
    geom_line(aes(y = season_adjust), color = "red") +
    ggtitle("VAR Actual (black) vs seasonally adjusted (red)")+
    xlab("Actual") +
    ylab("Seasonally adjusted value")
  
  VAR_Bagged_forecasts <- sim %>% 
    model(
      VAR1 = VAR(.sim),
    ) %>% 
    forecast(h = 3)
  
  Bagged_Summary <- VAR_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# Linear Ensemble Results -------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble_Linear"){

Best_Model <- Time_Series %>% 
  model(
    Linear1 = TSLM(Value ~ season() + trend()),
    Linear2 = TSLM(Value),
    Linear3 = TSLM(Value ~ season()),
    Linear4 = TSLM(Value ~ trend())
  ) %>% 
  mutate(Ensemble = (Linear1 + Linear2 + Linear3 + Linear4)/4)

Linear_Ensemble_Forecast <- Linear_Ensemble_fit %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
  group_by(Date, .model) %>% 
  summarise(
    dist = distributional::dist_sample(list(.sim))
  ) %>% 
  ungroup() %>% 
  as_fable(index = Date, key = .model, distribution = dist, response = "Value")

Best_Forecast_Plot <- Linear_Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Linear ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")

Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Linear ensemble innovation residuals")

Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Linear ensemble Autocorrelation function")

Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Linear Ensemble histogram of residuals")

Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Linear Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")

Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Linear Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")

Linear_Ensemble_forecast <- sim %>% 
  model(
    Linear1 = TSLM(sim ~ season() + trend()),
    Linear2 = TSLM(sim),
    Linear3 = TSLM(sim ~ season()),
    Linear4 = TSLM(sim ~ trend())
  ) %>% 
  mutate(Ensemble = (Linear1 + Linear2 + Linear3 + Linear4)/4) %>% 
  forecast(h = 3)

Bagged_Summary <- Linear_Ensemble_Bagged_forecasts %>% 
  summarise(bagged_mean = mean(.mean))
}

# Arima Ensemble Results --------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble_Arima"){

Best_Model <- Time_Series %>% 
  model(
    Arima1 = ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima2 = ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima3 = ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima4 = ARIMA(Value),
    Deterministic = ARIMA(Value ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Stochastic = ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
  ) %>% 
  mutate(Ensemble = (Arima1 + Arima2 + Arima3 + Arima4 + Deterministic + Stochastic)/6)

Arima_Ensemble_Forecast <- Best_Model %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
  group_by(Date, .model) %>% 
  summarise(
    dist = distributional::dist_sample(list(.sim))
  ) %>% 
  ungroup() %>% 
  as_fable(index = Date, key = .model, distribution = dist, response = "Value")

Best_Forecast_Plot <- Arima_Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Arima ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")

Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Arima ensemble innovation residuals")

Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Arima ensemble Autocorrelation function")

Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Arima Ensemble histogram of residuals")

Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Arima Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")

Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Arima Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")

Arima_Ensemble_Bagged_forecast <- sim %>% 
  model(
    Arima1 = ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima2 = ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima3 = ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima4 = ARIMA(Value),
    Deterministic = ARIMA(Value ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Stochastic = ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
  ) %>% 
  mutate(Ensemble = (Arima1 + Arima2 + Arima3 + Arima4 + Arima5 + Arima6 + Deterministic + Stochastic) / 6
  ) %>% 
  forecast(h = 3)

Bagged_Summary <- Arima_Ensemble_Bagged_forecasts %>% 
  summarise(bagged_mean = mean(.mean))
}

# ETS Ensemble Results ----------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble_Ets"){

Best_Model <- Time_Series %>% 
  model(
    Ets1 = ETS(Value ~ season() + trend()),
    Ets2 = ETS(Value ~ trend()),
    Ets3 = ETS(Value ~ season()),
    Ets4 = ETS(Value),
  ) %>% 
  mutate(Ensemble = (Ets1 + Ets2 + Ets3 + Ets4)/4)

Ets_Ensemble_Forecast <- Best_Model %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
  group_by(Date, .model) %>% 
  summarise(
    dist = distributional::dist_sample(list(.sim))
  ) %>% 
  ungroup() %>% 
  as_fable(index = Date, key = .model, distribution = dist, response = "Value")

Best_Forecast_Plot <- Ets_Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Ets ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")

Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Ets ensemble innovation residuals")

Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Ets ensemble Autocorrelation function")

Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Ets Ensemble histogram of residuals")

Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Ets Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")

Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Ets Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")

Ets_Ensemble_Bagged_forecast <- sim %>% 
  model(
    Ets1 = ETS(sim ~ season() + trend()),
    Ets2 = ETS(sim ~ trend()),
    Ets3 = ETS(sim ~ season()),
    Ets4 = ETS(sim),
  ) %>% 
  mutate(Ensemble = (Ets1 + Ets2 + Ets3 + Ets4)/4) %>% 
  forecast(h = 3)

Bagged_Summary <- Ets_Ensemble_Bagged_forecasts %>% 
  summarise(bagged_mean = mean(.mean))
}

# Holt-Winters Ensemble Results -------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble_Holt_Winters"){

Best_Model <- Time_Series %>% 
  model(
    Holt_Winters_Additive = ETS(Value ~ error("A") + trend("A") + season("A")),
    Holt_Winters_Multiplicative = ETS(Value ~ error("M") + trend("A") + season("M")),
    Holt_Winters_Damped = ETS(Value ~ error("M") + trend("Ad") + season("M")),
  ) %>% 
  mutate(Ensemble = (Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped) / 3)
         
Holt_Winters_Ensemble_Forecast <- Best_Model %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
    group_by(Date, .model) %>% 
    summarise(
    dist = distributional::dist_sample(list(.sim))
    ) %>% 
    ungroup() %>% 
    as_fable(index = Date, key = .model, distribution = dist, response = "Value")
         
Best_Forecast_Plot <- Holt_Winters_Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Holt-Winters ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")
         
Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Holt-Winters ensemble innovation residuals")
         
Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Holt-Winters ensemble Autocorrelation function")
         
Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Holt-Winters Ensemble histogram of residuals")
         
Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Holt-Winters Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")
         
Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Holt-Winters Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")
         
Holt_Winters_Ensemble_Bagged_forecast <- sim %>% 
  model(
  Holt_Winters_Additive = ETS(sim ~ error("A") + trend("A") + season("A")),
  Holt_Winters_Multiplicative = ETS(sim ~ error("M") + trend("A") + season("M")),
  Holt_Winters_Damped = ETS(sim ~ error("M") + trend("Ad") + season("M")),
  ) %>% 
  mutate(Ensemble = (Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped)/3)
    forecast(h = 3)
                  
Bagged_Summary <- Holt_Winters_Ensemble_Bagged_forecasts %>% 
  summarise(bagged_mean = mean(.mean))
}


# Fourier Ensemble Results ------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble_Holt_Fourier"){

Best_Model <- Time_Series %>% 
  model(
    Fourier1 = ARIMA(log(Value) ~ fourier(K = 1) + PDQ(0,0,0)),
    Fourier2 = ARIMA(log(Value) ~ fourier(K = 2) + PDQ(0,0,0)),
    Fourier3 = ARIMA(log(Value) ~ fourier(K = 3) + PDQ(0,0,0)),
    Fourier4 = ARIMA(log(Value) ~ fourier(K = 4) + PDQ(0,0,0)),
    Fourier5 = ARIMA(log(Value) ~ fourier(K = 5) + PDQ(0,0,0)),
    Fourier6 = ARIMA(log(Value) ~ fourier(K = 6) + PDQ(0,0,0)),
  ) %>% 
  mutate(Ensemble = (Fourier1 + Fourier2 + Fourier3 + Fourier4 + Fourier5 + Fourier6)/6)

Fourier_Ensemble_Forecast <- Best_Model %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
  group_by(Date, .model) %>% 
  summarise(
    dist = distributional::dist_sample(list(.sim))
  ) %>% 
  ungroup() %>% 
  as_fable(index = Date, key = .model, distribution = dist, response = "Value")

Best_Forecast_Plot <- Holt_Winters_Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Fourier ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")

Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Fourier ensemble innovation residuals")

Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Fourier ensemble Autocorrelation function")

Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Fourier Ensemble histogram of residuals")

Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Fourier Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")

Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Fourier Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")

Fourier_Ensemble_Bagged_forecast <- sim %>% 
  model(
    Holt_Winters_Additive = ETS(sim ~ error("A") + trend("A") + season("A")),
    Holt_Winters_Multiplicative = ETS(sim ~ error("M") + trend("A") + season("M")),
    Holt_Winters_Damped = ETS(sim ~ error("M") + trend("Ad") + season("M")),
  ) %>% 
  mutate(Ensemble = (Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped)/3) %>% 
  forecast(h = 3)
         
Bagged_Summary <- Fourier_Ensemble_Bagged_forecasts %>% 
  summarise(bagged_mean = mean(.mean))
}


# Ensemble Of All Time Series Models Results --------------------------------------------------------

if(Value_forecast_accuracy[1,1] == "Ensemble"){

Best_Model <- Time_Series %>% 
  model(
    Linear1 = TSLM(Value ~ season() + trend()),
    Linear2 = TSLM(Value),
    Linear3 = TSLM(Value ~ season()),
    Linear4 = TSLM(Value ~ trend()),
    Arima1 = ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima2 = ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima3 = ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima4 = ARIMA(Value),
    Deterministic = ARIMA(Value ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Stochastic = ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Ets1 = ETS(Value ~ season() + trend()),
    Ets2 = ETS(Value ~ trend()),
    Ets3 = ETS(Value ~ season()),
    Ets4 = ETS(Value),
    Holt_Winters_Additive = ETS(Value ~ error("A") + trend("A") + season("A")),
    Holt_Winters_Multiplicative = ETS(Value ~ error("M") + trend("A") + season("M")),
    Holt_Winters_Damped = ETS(Value ~ error("M") + trend("Ad") + season("M")),
    Fourier1 = ARIMA(log(Value) ~ fourier(K = 1) + PDQ(0,0,0)),
    Fourier2 = ARIMA(log(Value) ~ fourier(K = 2) + PDQ(0,0,0)),
    Fourier3 = ARIMA(log(Value) ~ fourier(K = 3) + PDQ(0,0,0)),
    Fourier4 = ARIMA(log(Value) ~ fourier(K = 4) + PDQ(0,0,0)),
    Fourier5 = ARIMA(log(Value) ~ fourier(K = 5) + PDQ(0,0,0)),
    Fourier6 = ARIMA(log(Value) ~ fourier(K = 6) + PDQ(0,0,0)),
    Prophet_Additive = fable.prophet::prophet(Value ~ season(period = 12, type = "additive")),
    Prophet_Multiplicative = fable.prophet::prophet(Value ~ season(period = 12, type = "multiplicative")),
    NeuralNet1 = fable::NNETAR(Value),
    NeuralNet2 = fable::NNETAR(Value ~ season()),
    NeuralNet3 = fable::NNETAR(Value ~ trend()),
    NeuralNet4 = fable::NNETAR(Value ~ season() + trend()),
    VAR1 = VAR(Value),
    Mean = fable::MEAN(Value),
    Naive = fable::NAIVE(Value),
    SNaive = fable::SNAIVE(Value),
    Drift = fable::SNAIVE(Value ~ drift())
  ) %>% 
  mutate(
    Ensemble = (Linear1 + Linear2 + Linear3 + Linear4 + Arima1 + Arima2 + Arima3 + Arima4 + Deterministic + Stochastic +
                  Ets1 + Ets2 + Ets3 + Ets4 + Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped +
                  Fourier1 + Fourier2 + Fourier3 + Fourier4 + Fourier5 + Fourier6 + Prophet_Additive + Prophet_Multiplicative +
                  NeuralNet1 + NeuralNet2 + NeuralNet3 + NeuralNet4 + VAR1 + Mean + Naive + SNaive + Drift) / 34
  )
    
Ensemble_Forecast <- Best_Model %>% 
  generate(h = 3, times = 100) %>% 
  as_tibble() %>% 
  group_by(Date, .model) %>% 
  summarise(
  dist = distributional::dist_sample(list(.sim))
  ) %>% 
  ungroup() %>% 
  as_fable(index = Date, key = .model, distribution = dist, response = "Value")
    
Best_Forecast_Plot <- Ensemble_Forecast %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(Time_Series) +
  labs(title = "Ensemble forecast") +
  scale_y_continuous(labels = scales::comma) +
  ylab("Value")
    
Best_Innovation_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  autoplot(.innov) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Ensemble innovation residuals")
    
Best_ACF <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ACF(.innov) %>% 
  autoplot() +
  labs(title = "Ensemble Autocorrelation function")
    
Best_Histogram_of_Residuals <- augment(Best_Model) %>% 
  filter(.model == "Ensemble") %>% 
  ggplot(aes(x = .resid)) +
  geom_histogram(bins = round(nrow(Time_Series)/5)) +
  geom_vline(xintercept = 0, color = "red") +
  labs(title = "Ensemble histogram of residuals")
    
Best_Actual_vs_Predicted <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .fitted)) +
  geom_point() +
  ggtitle("Ensemble Actual vs Predicted") +
  geom_abline(slope = 1, intercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Predicted")
    
Best_Actual_vs_Residuals <- 
  ggplot(augment(Best_Model) %>% filter(.model == "Ensemble"), mapping = aes(x = Value, y = .resid)) +
  geom_point() +
  ggtitle("Ensemble Actual vs Residuals") +
  geom_hline(yintercept = 0, color = "red")+
  xlab("Actual") +
  ylab("Residuals")
    
Ensemble_Bagged_forecast <- sim %>% 
  model(
    Linear1 = TSLM(sim ~ season() + trend()),
    Linear2 = TSLM(sim),
    Linear3 = TSLM(sim ~ season()),
    Linear4 = TSLM(sim ~ trend()),
    Arima1 = ARIMA(sim ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima2 = ARIMA(sim ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima3 = ARIMA(sim ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Arima4 = ARIMA(sim),
    Deterministic = ARIMA(sim ~  1 + pdq(d = 0), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Stochastic = ARIMA(sim ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE),
    Ets1 = ETS(sim ~ season() + trend()),
    Ets2 = ETS(sim ~ trend()),
    Ets3 = ETS(sim ~ season()),
    Ets4 = ETS(sim),
    Holt_Winters_Additive = ETS(sim ~ error("A") + trend("A") + season("A")),
    Holt_Winters_Multiplicative = ETS(sim ~ error("M") + trend("A") + season("M")),
    Holt_Winters_Damped = ETS(sim ~ error("M") + trend("Ad") + season("M")),
    Fourier1 = ARIMA(log(sim) ~ fourier(K = 1) + PDQ(0,0,0)),
    Fourier2 = ARIMA(log(sim) ~ fourier(K = 2) + PDQ(0,0,0)),
    Fourier3 = ARIMA(log(sim) ~ fourier(K = 3) + PDQ(0,0,0)),
    Fourier4 = ARIMA(log(sim) ~ fourier(K = 4) + PDQ(0,0,0)),
    Fourier5 = ARIMA(log(sim) ~ fourier(K = 5) + PDQ(0,0,0)),
    Fourier6 = ARIMA(log(sim) ~ fourier(K = 6) + PDQ(0,0,0)),
    Prophet_Additive = fable.prophet::prophet(sim ~ season(period = 12, type = "additive")),
    Prophet_Multiplicative = fable.prophet::prophet(sim ~ season(period = 12, type = "multiplicative")),
    NeuralNet1 = fable::NNETAR(sim),
    NeuralNet2 = fable::NNETAR(sim ~ season()),
    NeuralNet3 = fable::NNETAR(sim ~ trend()),
    NeuralNet4 = fable::NNETAR(sim ~ season() + trend()),
    VAR1 = VAR(sim),
    Mean = fable::MEAN(sim),
    Naive = fable::NAIVE(sim),
    SNaive = fable::SNAIVE(sim),
    Drift = fable::SNAIVE(sim ~ drift())
    ) %>% 
    mutate(
    Ensemble = (Linear1 + Linear2 + Linear3 + Linear4 + Arima1 + Arima2 + Arima3 + Arima4 + Deterministic + Stochastic +
      Ets1 + Ets2 + Ets3 + Ets4 + Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped +
      Fourier1 + Fourier2 + Fourier3 + Fourier4 + Fourier5 + Fourier6 + Prophet_Additive + Prophet_Multiplicative +
      NeuralNet1 + NeuralNet2 + NeuralNet3 + NeuralNet4 + VAR1 + Mean + Naive + SNaive + Drift) / 34
      ) %>% 
      mutate(Ensemble = (Holt_Winters_Additive + Holt_Winters_Multiplicative + Holt_Winters_Damped)/3)
        forecast(h = 3)

Bagged_Summary <- Ensemble_Bagged_forecasts %>% 
    summarise(bagged_mean = mean(.mean))
}

# 7. Summary Results for the model with lowest RMSE -----------------------

Best_Model

gt::gt(Value_forecast_accuracy, caption = "Time series forecast accuracy, sorted by Root Mean Squared Error (RMSE)")

Best_Forecast %>% 
  gt() %>% 
  fmt_number(columns = .mean, use_seps = TRUE) %>% 
  tab_header(title = "Best forecast")

Bagged_Summary %>% 
  gt() %>% 
  fmt_number(columns = bagged_mean, use_seps = TRUE) %>% 
  tab_header(title = "Bagged forecasts")

Best_Forecast_plot
Best_Forecast_plot

Best_Innovation_Residuals

Best_ACF

Best_Histogram_of_Residuals

Best_Actual_vs_Predicted 

Best_Actual_vs_Residuals

Best_Actual_vs_Trend

Best_Actual_vs_Seasonally_Adjusted
