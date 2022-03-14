#import packages
source("R/utils.R")
source("R/packages.R")


# Data prep -------

timeseries_obj <- read_rds("exam/hackathon_dataset.rds")
data_ts <- timeseries_obj$data

data_prep  <- data_ts %>%
  filter(period=="Daily") %>%
  group_by(id) %>%
  mutate(target = log_interval_vec(value)) %>%
  mutate(target = standardize_vec(target)) %>%
  select(-value, -period) %>%
  ungroup() %>%
  mutate(id = id %>% as.factor()) 


# * Train / Test Sets -----------------------------------------------------

data.training <- data_prep %>%
  filter(type=="train") %>%
  select(-type)
data.testing <- data_prep %>%
  filter(type=="test")%>%
  select(-type)

# * Recipes ---------------------------------------------------------------

rcp_spec <- recipe(target ~ ., data = data.training) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(id, fn = droplevels) %>%
  step_timeseries_signature(date) %>%
  #step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_rm(date) %>%
  step_zv(all_predictors())
rcp_spec %>% prep() %>% juice() %>% glimpse()

rcp_spec_spline <- rcp_spec %>%
  step_ns(ends_with("index.num"), deg_free = 2)
rcp_spec_spline %>% prep() %>% juice() %>% glimpse()

# Global Modelling Workflow -----------------------------------------------

# * linear regression ---------------------------------------------------------------
model_spec_lm <- linear_reg() %>%
  set_engine("lm")

# workflow fit - base_recipe
wrkfl_fit_lm <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec) %>%
  fit(data.training)

#workflow fit - spline recipe
wrkfl_fit_lm_spline <- workflow() %>%
  add_model(model_spec_lm) %>%
  add_recipe(rcp_spec_spline) %>%
  fit(data.training)

# * ELASTIC NET -----------------------------------------------------------

resamples_kfold <- data.training %>% vfold_cv(v = 3, repeats = 1)

model_spec_elanet <- linear_reg(
  mode = "regression",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

#workflow base
wrkl_elnet <- workflow() %>%
  add_model(model_spec_elanet) %>%
  add_recipe(rcp_spec)

#workflow spline
wrkl_elnet_spline <-  workflow() %>%
  add_model(model_spec_elanet) %>%
  add_recipe(rcp_spec_spline)



#tune spline
set.seed(42)
tune_res_elanet_spline <- tune_grid(
  object     = wrkl_elnet_spline,
  resamples  = resamples_kfold,
  param_info = parameters(wrkl_elnet_spline),
  grid       = 5,
  control    = control_grid(verbose = TRUE, allow_par = FALSE)
)


#tune base
set.seed(42)
tune_res_elanet <- tune_grid(
  object     = wrkl_elnet,
  resamples  = resamples_kfold,
  param_info = parameters(wrkl_elnet),
  grid       = 5,
  control    = control_grid(verbose = TRUE, allow_par = FALSE)
)



elnet_tuned_best_spline <- tune_res_elanet_spline %>%
  select_best("rmse")

elnet_tuned_best <- tune_res_elanet %>%
  select_best("rmse")


fin_wflw_elnet_spline <- wrkl_elnet_spline %>%
  finalize_workflow(parameters = elnet_tuned_best_spline)

fin_wflw_elnet <- wrkl_elnet %>%
  finalize_workflow(parameters = elnet_tuned_best)


wflw_fit_elnet_spline <- fin_wflw_elnet_spline %>%
  fit(data.training)


wflw_fit_elnet <- fin_wflw_elnet %>%
  fit(data.training)


# * BOOSTING --------------------------------------------------------------

model_spec_xgboost <- boost_tree(
  mode           = "regression",
  trees          = 500,
  tree_depth     = 3,
  learn_rate     = 0.1,
  stop_iter = 30
) %>%
  set_engine("xgboost")

wflw_fit_xgb <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(rcp_spec) %>%
  fit(data.training)

wflw_fit_xgb_spline <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(rcp_spec_spline) %>%
  fit(data.training)


# Calibration, Evaluation & Plotting --------------------------------------

calibration_tbl <- modeltime_table(
  wrkfl_fit_lm,
  wrkfl_fit_lm_spline,
  wflw_fit_elnet,
  wflw_fit_elnet_spline,
  wflw_fit_xgb,
  wflw_fit_xgb_spline
) %>%
  update_modeltime_description(.model_id = 1, .new_model_desc = "Linear regression") %>%
  update_modeltime_description(.model_id = 2, .new_model_desc = "Linear regression - spline") %>%
  update_modeltime_description(.model_id = 3, .new_model_desc = "Elastic net") %>%
  update_modeltime_description(.model_id = 4, .new_model_desc = "Elastic net - spline") %>%
  update_modeltime_description(.model_id = 5, .new_model_desc = "XGBoost") %>%
  update_modeltime_description(.model_id = 6, .new_model_desc = "XGBoost - spline") %>%
  modeltime_calibrate(data.testing, id = "id")

# Global accuracy
calibration_tbl %>%
  modeltime_accuracy(metric_set = metric_set(mape, rmse, rsq)) %>%
  table_modeltime_accuracy(.interactive = TRUE, bordered = TRUE, resizable = TRUE)

# Local accuracy
calibration_tbl %>%
  modeltime_accuracy(acc_by_id = TRUE) 

#Best model for each time series
gloabl_best_model_error <- calibration_tbl %>%
  modeltime_accuracy(acc_by_id=TRUE) %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() 

#table
calibration_tbl %>%
  modeltime_accuracy(acc_by_id=TRUE, metric_set = metric_set(mape, rmse, rsq)) %>%
  group_by(id) %>%
  slice_min(n=1, rmse) %>%
  ungroup() %>%
  table_modeltime_accuracy(.interactive = TRUE, bordered = TRUE, resizable = TRUE) 


#mean of RMSE, MASE & SMAPE for best models
mean(gloabl_best_model_error$rmse) 
mean(gloabl_best_model_error$mase)
mean(gloabl_best_model_error$smape)

############################################ENSEMBLE#################################################################
model_id_sel1 <- calibration_tbl %>%
  modeltime_accuracy() %>%
  arrange(rmse) %>%
  pull(.model_id)

submodels_sel1_tbl <- calibration_tbl %>%
  filter(.model_id %in% model_id_sel1)
submodels_sel1_tbl
# Loadings
loadings_sel1_tbl <- submodels_sel1_tbl %>%
  modeltime_accuracy() %>%
  mutate(rank = min_rank(-rmse)) %>%
  select(.model_id, rank)

# Fitting
ensemble_fit_wt_sel1 <- submodels_sel1_tbl %>%
  ensemble_weighted(loadings = loadings_sel1_tbl$rank)
ensemble_fit_wt_sel1$fit$loadings_tbl

calibration_tbl <- modeltime_table(
  wrkfl_fit_lm,
  wrkfl_fit_lm_spline,
  wflw_fit_elnet,
  wflw_fit_elnet_spline,
  wflw_fit_xgb,
  wflw_fit_xgb_spline,
  ensemble_fit_wt_sel1
) %>%
  update_modeltime_description(.model_id = 1, .new_model_desc = "Linear regression") %>%
  update_modeltime_description(.model_id = 2, .new_model_desc = "Linear regression - spline") %>%
  update_modeltime_description(.model_id = 3, .new_model_desc = "Elastic net") %>%
  update_modeltime_description(.model_id = 4, .new_model_desc = "Elastic net - spline") %>%
  update_modeltime_description(.model_id = 5, .new_model_desc = "XGBoost") %>%
  update_modeltime_description(.model_id = 6, .new_model_desc = "XGBoost - spline") %>%
  update_modeltime_description(.model_id = 7, .new_model_desc = "Ensemble") %>%
  modeltime_calibrate(data.testing, id = "id")

# Global accuracy
calibration_tbl %>%
  modeltime_accuracy(metric_set = metric_set(mase, rmse, smape)) %>%
  table_modeltime_accuracy(.interactive = TRUE, bordered = TRUE, resizable = TRUE)





