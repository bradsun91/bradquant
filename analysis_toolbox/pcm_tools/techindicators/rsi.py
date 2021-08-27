from pcm_tools.techindicators.ema import series_ema

def rsi(series, window, upper_threshold, lower_shreshold):

	window_length = window

	delta = series.diff()

	# Get rid of the first row, which is NaN since it did not have a previous 
	# row to calculate the differences
	delta = delta[1:] 

	# Make the positive gains (up) and negative gains (down) Series
	up, down = delta.copy(), delta.copy()
	up[up < 0] = 0
	down[down > 0] = 0

	# Calculate the EWMA
	roll_up1 = series_ema(up, window_length)
	roll_down1 = series_ema(down.abs(), window_length)

	# Calculate the RSI based on EWMA
	RS1 = roll_up1 / roll_down1
	RSI1 = 100.0 - (100.0 / (1.0 + RS1))

	return RSI1