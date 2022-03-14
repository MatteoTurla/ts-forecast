#import packages
source("R/utils.R")
source("R/packages.R")


# Data prep -------

timeseries_obj <- read_rds("exam/hackathon_dataset.rds")
data_ts <- timeseries_obj$data

daily_data <- data_ts[data_ts$period == "Weekly",]

daily_data %>% count(id, type)


daily_data_prep  <- daily_data %>%
  group_by(id) %>%
  mutate(target = log_interval_vec(value)) %>%
  mutate(target = standardize_vec(target)) %>%
  select(-type, -period, -value)

horizon <- 13
lag_period <- 13
rolling_periods <- c(13, 26)

data_prep_full_tbl <- daily_data_prep %>%
  extend_timeseries(
    .id_var = id,
    .date_var = date,
    .length_future = horizon
  ) %>%
  group_by(id) %>%
  # Add lags
  tk_augment_lags(target, .lags = lag_period) %>%
  # Add rolling features
  tk_augment_slidify(
    target_lag13,
    mean,
    .period = rolling_periods,
    .align = "center",
    .partial = TRUE
  ) %>%
  tk_augment_differences(.value = target_lag13,.lags = 1,.differences = 1) %>%
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .)) %>%
  ungroup()

nested_data_tbl <- data_prep_full_tbl %>%
  # Split into actual data & forecast data
  nest_timeseries(.id_var = id, .length_future = horizon)

nested_data_tbl <- nested_data_tbl %>%
  split_nested_timeseries(.length_test = horizon)

extract_nested_train_split(nested_data_tbl, .row_id = 1)
extract_nested_test_split(nested_data_tbl, .row_id = 1)

#  Recipes ---------------------------------------------------------------

rcp_naive <- 
  recipe(target ~ date, data = extract_nested_train_split(nested_data_tbl))
# Baseline Recipe
rcp_spec <-
  # recipe(optins_trans ~ ., data = training(splits)) %>%
  recipe(target ~ ., data = extract_nested_train_split(nested_data_tbl)) %>%
  # Time Series Signature
  step_timeseries_signature(date) %>%
  step_rm(matches("(iso)|(xts)|(minute)|(second)|(date_wday)|(date_wday)|(date_am.pm)|(date_hour)|(date_hour12)|(date_day)")) %>%
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  # Interaction
  # Fourier
  step_fourier(date, period = c(4, 8, 24, 48), K = 2) %>%
  step_zv(all_predictors())
rcp_spec %>% prep() %>% juice() %>% glimpse()

# Spline Recipe
# - natural spline series on index.num
rcp_spec_spline <- rcp_spec %>%
  step_ns(ends_with("index.num"), deg_free = 2) %>%
  step_rm(date) %>%
  step_rm(starts_with("lag_"))
rcp_spec_spline %>% prep() %>% juice() %>% glimpse()

# Lag Recipe
# - lags of optins_trans and rolls
rcp_spec_lag <- rcp_spec %>%
  step_naomit(starts_with("lag_")) %>%
  step_rm(date)
rcp_spec_lag %>% prep() %>% juice() %>% glimpse()

# Naive ---------

# NAIVE
model_spec_naive <- naive_reg() %>%
  set_engine("naive") 


# WINDOW - MEAN
model_spec_mean <- window_reg(
  window_size = 4
) %>%
  set_engine(
    "window_function",
    window_function = mean,
    na.rm = TRUE
  ) 

# WINDOW - WEIGHTED MEAN
model_spec_wmean <- window_reg(
  window_size = 4
) %>%
  set_engine(
    "window_function",
    window_function = ~ sum(tail(.x, 3) * c(0.1, 0.3, 0.6))
  ) 
# WINDOW - MEDIAN
model_spec_median <- window_reg(
  window_size = 4
) %>%
  set_engine(
    "window_function",
    window_function = median,
    na.rm = TRUE
  ) 

wrkfl_fit_naive <- workflow() %>%
  add_model(model_spec_naive) %>%
  add_recipe(rcp_naive)

wrkfl_fit_mean <- workflow() %>%
  add_model(model_spec_mean) %>%
  add_recipe(rcp_naive)

wrkfl_fit_wmean <- workflow() %>%
  add_model(model_spec_naive) %>%
  add_recipe(rcp_naive)

wrkfl_fit_median <- workflow() %>%
  add_model(model_spec_median) %>%
  add_recipe(rcp_naive)


nested_modeltime_tbl_naive <- nested_data_tbl %>%
  modeltime_nested_fit(
    model_list = list(
      wrkfl_fit_naive,
      wrkfl_fit_mean,
      wrkfl_fit_wmean,
      wrkfl_fit_median
    ),
    control = control_nested_fit(
      verbose = TRUE,
      allow_par = FALSE
    )
  )
# Accuracy
nested_modeltime_tbl_naive %>%
  extract_nested_test_accuracy() %>%
  table_modeltime_accuracy()

nested_modeltime_tbl_naive %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() %>%
  table_modeltime_accuracy()

nested_best_model_error_naive <- nested_modeltime_tbl_naive %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() 
 

mean(nested_best_model_error_naive$rmse) 
mean(nested_best_model_error_naive$mase)
mean(nested_best_model_error_naive$smape)




# S-ARIMA-X ---------------------------------------------------------------

# Auto-SARIMA
model_spec_auto_sarima <- arima_reg() %>%
  set_engine("auto_arima") 

wrkfl_auto_sarima <- workflow() %>%
  add_recipe(rcp_naive) %>%
  add_model(model_spec_auto_sarima) 



# PROPHET -----------------------------------------------------------------

# Auto-PROPHET
model_spec_auto_prophet <- prophet_reg() %>%
  set_engine("prophet") 

wrkfl_auto_prophet <- workflow() %>%
  add_recipe(rcp_naive) %>%
  add_model(model_spec_auto_prophet)


# Lr ----
model_spec_lm <- linear_reg() %>%
  set_engine("lm")

# LM + Splines
w_lm_spline <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec_spline)

# LM + Lags
w_lm_lag <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec_lag)

# XGB ---
model_spec_xgboost <- boost_tree(
  mode           = "regression",
  trees          = 500,
  tree_depth     = 3,
  learn_rate     = 0.03,
) %>%
  set_engine("xgboost")

w_xgb_lag <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(rcp_spec_lag)

w_xgb_spline <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(rcp_spec_spline)

# prophet xgb ----

model_spec_prophet_xgboost <- prophet_boost(
  mode           = "regression",
  trees          = 500,
  tree_depth     = 3,
  learn_rate     = 0.03,
) %>%
  set_engine("prophet_xgboost")

w_prophet_xgb <- workflow() %>%
  add_model(model_spec_prophet_xgboost) %>%
  add_recipe(rcp_spec)

nested_modeltime_tbl <- nested_data_tbl %>%
  modeltime_nested_fit(
    model_list = list(
      wrkfl_auto_sarima,
      wrkfl_auto_prophet, 
      w_lm_lag, 
      w_lm_spline,
      w_xgb_lag, 
      w_xgb_spline,
      w_prophet_xgb
    ),
    control = control_nested_fit(
      verbose = TRUE,
      allow_par = FALSE
    )
  )

# Accuracy
nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  table_modeltime_accuracy()

nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() %>%
  table_modeltime_accuracy()

nested_best_model_error <- nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() 

mean(nested_best_model_error$rmse) 
mean(nested_best_model_error$mase)
mean(nested_best_model_error$smape)

# Error reporting
nested_modeltime_tbl %>%
  extract_nested_error_report()