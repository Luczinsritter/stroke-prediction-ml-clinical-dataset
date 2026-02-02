from utils import threshold_f2_optimizer, threshold_layer, classification_report
from pipeline import  x_train, y_train, x_test, y_test
from model import trained_model

# Find the optimize value to tune the decision threshold of the model
f2_optimized_threshold, _ = threshold_f2_optimizer(trained_model, x_train, y_train)

# Running the model in the hard code layer to apply the threshold
y_pred = threshold_layer(trained_model, x_test, f2_optimized_threshold)

# Display the model performance
print(classification_report(y_test, y_pred))