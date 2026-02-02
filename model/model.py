from utils import RandomForestClassifier
from pipeline import x_train, y_train

# Setting and train the model
rfc_model = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=5, min_impurity_decrease=1e-4,
                                            min_samples_leaf=15, max_features='sqrt', class_weight='balanced_subsample',
                                            bootstrap=True, max_samples=0.75, ccp_alpha=0.0005, random_state=22, n_jobs=-1)
trained_model = rfc_model.fit(x_train, y_train)