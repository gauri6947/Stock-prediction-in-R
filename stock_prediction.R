# Load required libraries
library(quantmod)
library(caTools)
library(TTR)
library(xgboost)
library(ggplot2)

# Step 1: Get Stock Data
symbol <- "AMZN"
getSymbols(symbol, src = "yahoo", from = "2020-01-01", to = "2024-01-01")
stock_data <- Cl(get(symbol))

# Step 2: Feature Engineering
data <- data.frame(Date = index(stock_data), Price = as.numeric(stock_data))

# Technical Indicators
data$SMA20 <- SMA(data$Price, n = 20)
data$SMA50 <- SMA(data$Price, n = 50)
data$EMA10 <- EMA(data$Price, n = 10)
data$RSI14 <- RSI(data$Price, n = 14)
data$Volatility <- runSD(data$Price, n = 10)
data$Return <- ROC(data$Price, type = "discrete")

# MACD Calculation
macd_data <- MACD(data$Price, nFast = 12, nSlow = 26, nSig = 9)
data$MACD <- macd_data[, 1]          # MACD line
data$MACD_Signal <- macd_data[, 2]   # Signal line

# Ensure the price data is a time series for Bollinger Bands
price_ts <- as.xts(data$Price, order.by = data$Date)

# Bollinger Bands Calculation
bbands <- BBands(price_ts, n = 20, maType = "SMA")
data$BollingerHigh <- as.numeric(bbands[, "up"])
data$BollingerLow <- as.numeric(bbands[, "dn"])
data$BollingerMid <- as.numeric(bbands[, "mavg"])  # Middle Bollinger Band if needed

# Seasonality Features
data$DayOfWeek <- as.numeric(format(data$Date, "%u"))
data$Month <- as.numeric(format(data$Date, "%m"))

# Lags and Returns
data$Lag1 <- lag(data$Price, -1)
data$Lag2 <- lag(data$Price, -2)
data$Return_Lag1 <- lag(data$Return, -1)
data$Return_Lag5 <- lag(data$Return, -5)

# Remove NA values after feature creation
data <- na.omit(data)

# Step 3: Split Data into Training and Testing Sets
set.seed(123)
split <- sample.split(data$Price, SplitRatio = 0.8)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Step 4: Prepare Data for XGBoost
train_matrix <- model.matrix(Price ~ . - Date, data = train_data)
test_matrix <- model.matrix(Price ~ . - Date, data = test_data)

dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$Price)
dtest <- xgb.DMatrix(data = test_matrix, label = test_data$Price)

# Step 5: Train the XGBoost Model
params <- list(
  objective = "reg:squarederror",   # For regression
  eta = 0.1,                        # Learning rate
  max_depth = 6,                    # Depth of the trees
  subsample = 0.8,                  # Use 80% of data for each tree
  colsample_bytree = 0.8            # Use 80% of features for each tree
)

xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 150, watchlist = list(train = dtrain, test = dtest), print_every_n = 10)

# Step 6: Make Predictions
test_data$Predicted <- predict(xgb_model, dtest)

# Step 7: Plot Actual vs. Predicted Prices
ggplot() +
  geom_line(data = test_data, aes(x = Date, y = Price), color = "blue") +
  geom_line(data = test_data, aes(x = Date, y = Predicted), color = "red") +
  labs(title = paste(symbol, "Stock Price Prediction with XGBoost"),
       x = "Date",
       y = "Price") +
  theme_minimal() +
  scale_x_date(date_labels = "%b %Y", date_breaks = "6 months") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Step 8: Evaluate the Model
mse <- mean((test_data$Price - test_data$Predicted)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data$Price - test_data$Predicted))

cat("Mean Squared Error:", round(mse, 2), "\n")
cat("Root Mean Squared Error:", round(rmse, 2), "\n")
cat("Mean Absolute Error:", round(mae, 2), "\n")
